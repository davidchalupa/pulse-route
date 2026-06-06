import random
import requests
import folium
from folium.plugins import TimestampedGeoJson
import os
from datetime import datetime, timedelta
from shapely.geometry import shape, Point
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import pickle
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from math import radians, cos, sin, asin, sqrt

# Import original and new routing engines
import routing_engine_nn
import routing_engine_ls_2opt
import routing_engine_cvrp  # Core Engine for Global Optimization

# --- App Configuration & Styling ---
st.set_page_config(page_title="PulseRoute Simulator", page_icon="🚚", layout="wide")
st.title("🚚 PulseRoute Logistics Simulator")
st.markdown("Model demand, configure fleets, and visualize dynamic routing.")


# --- 1. Geospatial & Boundary Helpers ---
@st.cache_resource(show_spinner=False)
def get_city_data(city_name):
    """Fetches city boundary, center, and road network with caching."""
    cache_dir = "city_cache"
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = city_name.replace(",", "").replace(" ", "_").lower()
    cache_path = os.path.join(cache_dir, f"{safe_name}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    headers = {'User-Agent': 'PulseRouteSimulation_v4_Streamlit'}
    params = {'q': city_name, 'polygon_geojson': 1, 'format': 'json', 'limit': 1}
    url = "https://nominatim.openstreetmap.org/search"

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200 or not response.json():
        raise ValueError(f"Could not find or fetch data for '{city_name}'.")

    data = response.json()[0]
    center_coords = (float(data['lat']), float(data['lon']))
    boundary_shape = shape(data['geojson'])

    # Download graph
    graph = ox.graph_from_polygon(boundary_shape, network_type='drive')

    city_data = (center_coords, boundary_shape, graph)
    with open(cache_path, 'wb') as f:
        pickle.dump(city_data, f)

    return city_data


# --- 2. Enhanced Demand Generation ---
class Order:
    def __init__(self, order_id, coords, order_time, deadline):
        self.id = order_id
        self.coords = coords
        self.order_time = order_time
        self.deadline = deadline
        self.delivered_at = None
        self.assigned_vehicle = None


class DemandManager:
    DEMAND_LEVELS = {"Low (20)": 20, "Medium (50)": 50, "High (100)": 100}

    @classmethod
    def generate_realistic_demand(cls, city_polygon, start_time, num_orders, hourly_weights, tightness):
        centroid = city_polygon.centroid
        center_lat, center_lon = centroid.y, centroid.x
        orders = []
        attempts = 0

        while len(orders) < num_orders and attempts < 2000:
            lat = random.gauss(center_lat, tightness)
            lon = random.gauss(center_lon, tightness)
            pnt = Point(lon, lat)

            if city_polygon.contains(pnt):
                hour = random.choices(list(hourly_weights.keys()), weights=list(hourly_weights.values()))[0]
                minute = random.randint(0, 59)
                order_time = start_time.replace(hour=hour, minute=minute)
                deadline = order_time + timedelta(hours=2.5)
                orders.append(Order(f"ORD-{len(orders) + 1:03d}", (lat, lon), order_time, deadline))
            attempts += 1

        orders.sort(key=lambda o: o.order_time)
        return orders


# --- 3. Multi-Vehicle Simulation Logic ---
class DeliverySimulation:
    def __init__(self, depot_coords, orders, graph, num_vehicles=1, vehicle_speed_kmh=45,
                 max_wait_minutes=30, vehicle_capacity=5, route_algorithm=None,
                 optimization_scope="Local", sla_mode="hard"):
        self.depot = depot_coords
        self.orders = orders
        self.graph = graph
        self.vehicle_speed_kmh = vehicle_speed_kmh
        self.vehicle_speed_mps = vehicle_speed_kmh * (1000 / 3600)
        self.max_wait_minutes = max_wait_minutes
        self.vehicle_capacity = vehicle_capacity
        self.route_algorithm = route_algorithm
        self.optimization_scope = optimization_scope
        self.sla_mode = sla_mode

        # Initialize fleet
        self.vehicles = [{"id": i, "loc": depot_coords, "time": orders[0].order_time if orders else datetime.now(),
                          "trajectory": [], "distance": 0.0} for i in range(num_vehicles)]

    def _haversine(self, coord1, coord2):
        """Calculates the great-circle distance between two points in meters."""
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * asin(sqrt(a)) * 6371000

    def _calculate_theoretical_lower_bound(self):
        """Computes the 'Greedy Ideal' lower bound for routing distance."""
        if not self.orders:
            return 0.0

        n = len(self.orders)
        dist_matrix = np.zeros((n, n))
        coords = [o.coords for o in self.orders]

        # 1. Build Distance Matrix
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._haversine(coords[i], coords[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # 2. Compute Minimum Spanning Tree (MST)
        mst = minimum_spanning_tree(dist_matrix)
        mst_distance = mst.sum()

        # 3. Apply Network Circuity Factor (approx. 25% road overhead vs straight line)
        circuity_factor = 1.25
        network_mst_dist = mst_distance * circuity_factor

        # 4. Return to Depot Penalty (Approximated max cycles required)
        depot_distances = [self._haversine(self.depot, c) for c in coords]
        max_depot_dist = max(depot_distances)
        trips_required = n / self.vehicle_capacity
        depot_penalty = max_depot_dist * 2 * trips_required

        return network_mst_dist + depot_penalty

    def _get_road_route(self, start_coords, end_coords):
        orig_node = ox.nearest_nodes(self.graph, start_coords[1], start_coords[0])
        dest_node = ox.nearest_nodes(self.graph, end_coords[1], end_coords[0])
        try:
            route = nx.shortest_path(self.graph, orig_node, dest_node, weight='length')
            path_coords = []
            path_length = 0
            nodes_data = self.graph.nodes
            for i in range(len(route)):
                node = route[i]
                path_coords.append((nodes_data[node]['y'], nodes_data[node]['x']))
                if i > 0:
                    path_length += self.graph.get_edge_data(route[i - 1], route[i])[0]['length']
            return path_coords, path_length
        except nx.NetworkXNoPath:
            return [start_coords, end_coords], routing_engine_cvrp.calculate_haversine_distance(start_coords,
                                                                                                end_coords)

    def _travel(self, vehicle, end_loc):
        path_points, dist_meters = self._get_road_route(vehicle["loc"], end_loc)
        vehicle["distance"] += dist_meters
        travel_time_sec = dist_meters / self.vehicle_speed_mps

        if len(path_points) > 1:
            time_per_segment = travel_time_sec / (len(path_points) - 1)
            for i, p in enumerate(path_points):
                t = vehicle["time"] + timedelta(seconds=time_per_segment * i)
                vehicle["trajectory"].append((t, p))

        vehicle["time"] += timedelta(seconds=travel_time_sec)
        vehicle["loc"] = end_loc

    def run(self):
        on_time = 0
        theoretical_min_distance = self._calculate_theoretical_lower_bound()

        # ==========================================================
        # STRATEGY A: GLOBAL CVRP OPTIMIZATION (Trip-Aware Deployment)
        # ==========================================================
        if "Global" in self.optimization_scope:
            # Let the CVRP engine figure out fleet assignment and sequence upfront using trips
            global_trips = routing_engine_cvrp.global_cvrp_solver(
                self.depot, self.orders, len(self.vehicles), self.vehicle_capacity,
                self.vehicle_speed_kmh, sla_mode=self.sla_mode
            )

            for v in self.vehicles:
                assigned_trips = global_trips.get(v["id"], [])
                if not assigned_trips:
                    continue

                # Align vehicle starting clock safely to its first assigned trip
                if v["time"] < assigned_trips[0][0].order_time:
                    v["time"] = assigned_trips[0][0].order_time

                for trip in assigned_trips:
                    if not trip:
                        continue

                    # Log departure from depot
                    v["trajectory"].append((v["time"], v["loc"]))
                    v["time"] += timedelta(minutes=5)  # Loading time at depot for this trip

                    # Deliver the optimized sequence within this discrete trip
                    for order in trip:
                        # Safety fallback: vehicle clock cannot deliver an order before it is placed
                        if v["time"] < order.order_time:
                            v["time"] = order.order_time
                            v["trajectory"].append((v["time"], v["loc"]))

                        self._travel(v, order.coords)
                        order.delivered_at = v["time"]
                        order.assigned_vehicle = v["id"]
                        if order.delivered_at <= order.deadline:
                            on_time += 1
                        v["time"] += timedelta(minutes=2)  # Dropoff time

                    # Return home to depot to conclude this trip and prepare for next capacity cycle
                    self._travel(v, self.depot)

        # ==========================================================
        # STRATEGY B: LOCAL FIXED BATCHES (Original Sequential Constraint)
        # ==========================================================
        else:
            unassigned_orders = sorted(self.orders, key=lambda o: o.order_time)

            while unassigned_orders:
                # Pick the earliest available vehicle
                v = min(self.vehicles, key=lambda x: x["time"])

                if v["time"] < unassigned_orders[0].order_time:
                    v["time"] = unassigned_orders[0].order_time
                    v["trajectory"].append((v["time"], v["loc"]))

                dispatch_time = unassigned_orders[0].order_time + timedelta(minutes=self.max_wait_minutes)

                batch = []
                for o in list(unassigned_orders):
                    if o.order_time <= dispatch_time and len(batch) < self.vehicle_capacity:
                        batch.append(o)
                    elif len(batch) >= self.vehicle_capacity:
                        break

                v["time"] = max(v["time"], batch[-1].order_time) + timedelta(minutes=5)  # Load time

                for o in batch:
                    unassigned_orders.remove(o)

                # Execute route sequence optimization via selected local strategy wrapper
                try:
                    route_orders = self.route_algorithm(v["loc"], batch, route_id=f"Vehicle_{v['id']}_Batch")
                except TypeError:
                    route_orders = self.route_algorithm(v["loc"], batch)

                # Execute route travel simulation
                for order in route_orders:
                    if v["time"] < order.order_time:
                        v["time"] = order.order_time
                    self._travel(v, order.coords)
                    order.delivered_at = v["time"]
                    order.assigned_vehicle = v["id"]
                    if order.delivered_at <= order.deadline:
                        on_time += 1
                    v["time"] += timedelta(minutes=2)  # Dropoff time

                # Return to depot
                self._travel(v, self.depot)

        total_distance = sum(v["distance"] for v in self.vehicles)
        return {
            "orders": self.orders,
            "vehicles": self.vehicles,
            "on_time": on_time,
            "total_distance": total_distance,
            "theoretical_min_distance": theoretical_min_distance
        }


# --- 4. Stepper Logic & UI Pipeline ---

if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

step_names = ["📍 Location", "📊 Demand", "⚙️ Fleet", "🏁 Simulate"]

st.progress((st.session_state.current_step + 1) / len(step_names))
cols = st.columns(len(step_names))
for i, name in enumerate(step_names):
    if i == st.session_state.current_step:
        cols[i].markdown(f"**{name}**")
    else:
        cols[i].markdown(f"<span style='color: gray'>{name}</span>", unsafe_allow_html=True)

st.divider()

# --- STEP 1: LOCATION ---
if st.session_state.current_step == 0:
    st.header("Select Operating City")
    city_input = st.text_input("Enter City, Country", value="Bratislava, Slovakia")

    if st.button("Fetch Map Data"):
        with st.spinner("Fetching data and mapping road network (this may take a minute on first run)..."):
            try:
                st.session_state['city_data'] = get_city_data(city_input)
                st.success(f"Loaded {city_input} successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    if 'city_data' in st.session_state:
        depot_loc, boundary, _ = st.session_state['city_data']
        m = folium.Map(location=depot_loc, zoom_start=12)
        folium.GeoJson(boundary, style_function=lambda x: {'color': 'blue', 'fillOpacity': 0.1}).add_to(m)
        folium.Marker(depot_loc, popup="Depot (Center)").add_to(m)
        components.html(m._repr_html_(), height=400)


# --- STEP 2: DEMAND ---
elif st.session_state.current_step == 1:
    st.header("Generate Order Demand")
    if 'city_data' not in st.session_state:
        st.warning("Please go back and select a city first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            profile_type = st.selectbox("Demand Profile",
                                        ["Single Peak (Noon)", "Two Peaks (Noon & Evening)", "Custom Flat"])
            vol = st.selectbox("Volume", ["Low (20)", "Medium (50)", "High (100)"], index=1)
            num_orders = {"Low (20)": 20, "Medium (50)": 50, "High (100)": 100}[vol]
            tightness = st.slider("Customer Spread", 0.01, 0.08, 0.03,
                                  help="Lower = tightly clustered around depot. Higher = spread across city.")

        with col2:
            if profile_type == "Single Peak (Noon)":
                weights = {8: 1, 9: 2, 10: 3, 11: 5, 12: 15, 13: 8, 14: 4, 15: 2, 16: 1}
            elif profile_type == "Two Peaks (Noon & Evening)":
                weights = {8: 2, 9: 3, 10: 2, 11: 4, 12: 12, 13: 5, 14: 3, 15: 4, 16: 6, 17: 12, 18: 8, 19: 2}
            else:
                weights = {h: 5 for h in range(8, 18)}

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(weights.keys(), weights.values(), color='#3498db')
            ax.set_title("Probability Distribution of Orders")
            ax.set(xlabel="Hour of Day", ylabel="Weight")
            st.pyplot(fig)

        if st.button("Generate Demand"):
            start_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            _, boundary, _ = st.session_state['city_data']
            st.session_state['orders'] = DemandManager.generate_realistic_demand(
                boundary, start_date, num_orders, weights, tightness
            )
            st.success(f"Generated {num_orders} orders successfully!")

        # Display the generated demand points immediately on the map if they exist
        if 'orders' in st.session_state:
            st.markdown("### 📍 Generated Demand Visualization")
            depot_loc, boundary, _ = st.session_state['city_data']

            # Construct preview map
            preview_map = folium.Map(location=depot_loc, zoom_start=12, tiles="cartodbpositron")
            folium.GeoJson(boundary, style_function=lambda x: {'color': 'gray', 'fillOpacity': 0.05}).add_to(
                preview_map)
            folium.Marker(depot_loc, icon=folium.Icon(color='black', icon='home'), popup="Central Depot").add_to(
                preview_map)

            # Plot generated order demand locations as blue circles before simulation assignment
            for o in st.session_state['orders']:
                folium.CircleMarker(
                    location=o.coords, radius=4, color="#3498db", fill=True, fill_opacity=0.7,
                    popup=f"<b>Order:</b> {o.id}<br><b>Placed:</b> {o.order_time.strftime('%H:%M')}"
                ).add_to(preview_map)

            components.html(preview_map._repr_html_(), height=450)


# --- STEP 3: FLEET CONFIGURATION ---
elif st.session_state.current_step == 2:
    st.header("Fleet & Constraint Configuration")
    if 'orders' not in st.session_state:
        st.warning("Please generate demand first.")
    else:
        c1, c2 = st.columns(2)
        num_v = c1.number_input("Vehicles", min_value=1, max_value=10, value=2)
        cap = c1.number_input("Capacity", min_value=1, max_value=20, value=5)
        wait = c2.slider("Max Wait (mins)", 0, 60, 25,
                         help="How long a package waits at the depot to build a batch (Only applies to Local Strategy).")
        spd = c2.slider("Speed (km/h)", 20, 80, 45)

        st.session_state['sim_params'] = {
            "num_vehicles": num_v,
            "vehicle_capacity": cap,
            "max_wait_minutes": wait,
            "vehicle_speed_kmh": spd
        }
        st.success("Fleet configurations recorded! Proceed to simulation screen to select optimization engine.")


# --- STEP 4: SIMULATE ---
elif st.session_state.current_step == 3:
    st.header("Run Simulation & View Results")
    if 'sim_params' not in st.session_state:
        st.warning("Please configure fleet parameters first.")
    else:
        # --- Optimization Scope Selection (Moved to Simulation Page for On-Screen Alternating) ---
        st.markdown("### 🎛️ Optimization Scope Selection")

        opt_scope = st.selectbox(
            "Select System Optimization Strategy:",
            options=["Local (Fixed Sequential Batches)", "Global (Dynamic Fleet Assignment)"],
            help="Choose how orders are clustered and dispatched to your fleet."
        )

        # Dynamic Explainer Blocks based on selection
        if opt_scope == "Local (Fixed Sequential Batches)":
            st.info(
                "⏳ **How it works:** Orders are locked into chronological batches based on order time. "
                "Once a batch fills up or hits the Max Wait threshold, the system triggers local pathing "
                "(Nearest Neighbor/2-Opt) for that single batch.\n\n"
                "⚖️ **Trade-off:** Fast and predictable execution, but can result in **higher overall mileage** "
                "if concurrent orders are scattered on opposite sides of the city."
            )
        else:
            st.success(
                "🌍 **How it works:** Solves a global Capacitated Vehicle Routing Problem (CVRP). "
                "It treats the entire fleet and order pool as a holistic ecosystem, dynamically assignmenting "
                "trips and sequencing stops to maximize spatial efficiency.\n\n"
                "⚖️ **Trade-off:** Delivers **significantly lower total mileage** and optimized fleet utilization, "
                "but requires a heavier computational footprint."
            )

        chosen_algorithm = routing_engine_ls_2opt.base_route_sequencer
        engine_name_notice = "2-Opt Local Search Sequencer"
        sla_constraint_type = "hard"

        if opt_scope == "Local (Fixed Sequential Batches)":
            selected_engine_name = st.selectbox(
                "Local Routing Sequence Variant",
                options=["Nearest Neighbor Heuristic (Fast Baseline)", "2-Opt Local Search (Path Untangling)"],
                index=1
            )
            if "Nearest Neighbor" in selected_engine_name:
                chosen_algorithm = routing_engine_nn.base_route_sequencer
                engine_name_notice = "Nearest Neighbor Sequencer"
        else:
            sla_constraint_type = st.radio(
                "Global SLA Rule Enforcement:",
                options=["hard", "dynamic"],
                format_func=lambda
                    x: "Strict Constraint (Forces 100% SLA)" if x == "hard" else "Dynamic Balancing (Prioritizes Mileage Reduction)",
                help="Hard Constraints completely reject paths that violate deadlines. Dynamic allows minor lateness if it yields massive fuel/distance savings."
            )

        # --- Collapsible Strategy Comparison Summary Table Placeholder ---
        summary_placeholder = st.empty()
        with summary_placeholder.container():
            if 'sim_history' in st.session_state and st.session_state['sim_history']:
                with st.expander("📊 Collapsible Strategy Performance Comparison Summary", expanded=True):
                    st.table(st.session_state['sim_history'])

        if st.button("🚀 Start Simulation", type="primary"):
            depot_loc, boundary, graph = st.session_state['city_data']
            orders = st.session_state['orders']

            # Combine static fleet parameters with strategy selections made on-screen
            params = st.session_state['sim_params'].copy()
            params.update({
                "route_algorithm": chosen_algorithm,
                "optimization_scope": opt_scope,
                "sla_mode": sla_constraint_type
            })

            with st.spinner(f"Simulating routing via {params['optimization_scope']}..."):
                for o in orders:
                    o.delivered_at = None

                # Creates simulation and injects configuration params
                sim = DeliverySimulation(depot_loc, orders, graph, **params)
                results = sim.run()

                # Calculate run metrics
                success_rate = (results['on_time'] / len(orders)) * 100
                actual_distance_km = results['total_distance'] / 1000
                lower_bound_km = results['theoretical_min_distance'] / 1000
                efficiency_gap = (results['total_distance'] / results['theoretical_min_distance']) if results[
                                                                                                          'theoretical_min_distance'] > 0 else 1

                # Append metrics into summary logs matching specific names
                if 'sim_history' not in st.session_state:
                    st.session_state['sim_history'] = {}

                history_key = f"{opt_scope}"
                if opt_scope == "Local (Fixed Sequential Batches)":
                    history_key += f" ({'NN' if 'Nearest Neighbor' in selected_engine_name else '2-Opt'})"
                else:
                    history_key += f" ({sla_constraint_type.capitalize()} SLA)"

                st.session_state['sim_history'][history_key] = {
                    "SLA Success": f"{success_rate:.1f}%",
                    "Actual Route Dist": f"{actual_distance_km:.2f} km",
                    "Ideal Bound": f"{lower_bound_km:.2f} km",
                    "Overhead Factor": f"{efficiency_gap:.2f}x"
                }

                # Dynamically write updated statistics into the summary container block instantly on finish
                with summary_placeholder.container():
                    with st.expander("📊 Collapsible Strategy Performance Comparison Summary", expanded=True):
                        st.table(st.session_state['sim_history'])

                # --- Metrics Display ---
                st.subheader("Simulation Performance Analysis")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("SLA Success Rate", f"{success_rate:.1f}%",
                          f"{results['on_time']}/{len(orders)} On Time",
                          delta_color="normal" if success_rate > 90 else "inverse")
                c2.metric("Actual Distance", f"{actual_distance_km:.2f} km")

                # Compare actual vs theoretical ideal
                c3.metric("Theoretical Min (Greedy Ideal)", f"{lower_bound_km:.2f} km",
                          f"{efficiency_gap:.1f}x Multiplier", delta_color="off")

                c4.metric("Active Fleet", f"{params['num_vehicles']} Vehicles Used")

                # --- Leaflet Map Reconstruction Pipeline ---
                m = folium.Map(location=depot_loc, zoom_start=13, tiles="cartodbpositron")
                folium.GeoJson(boundary, style_function=lambda x: {'color': 'gray', 'fillOpacity': 0.05}).add_to(m)
                folium.Marker(depot_loc, icon=folium.Icon(color='black', icon='home'), popup="Central Depot").add_to(m)

                colors = ['#3498db', '#e74c3c', '#9b59b6', '#f1c40f', '#e67e22', '#2ecc71']
                features = []

                for v in results["vehicles"]:
                    coords, times = [], []
                    for t, point in v["trajectory"]:
                        coords.append([point[1], point[0]])
                        times.append(t.strftime('%Y-%m-%dT%H:%M:%S'))

                    if coords:
                        color = colors[v["id"] % len(colors)]
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {"times": times, "style": {"color": color, "weight": 4, "opacity": 0.7}}
                        })
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": coords[0]},
                            "properties": {"times": times, "icon": "circle",
                                           "iconstyle": {"fillColor": color, "fillOpacity": 1, "stroke": "true",
                                                         "radius": 7}}
                        })

                # Plot order drops matching original custom formatting structure
                for o in results["orders"]:
                    m_color = "green" if o.delivered_at and o.delivered_at <= o.deadline else "red"
                    deliv_time = o.delivered_at.strftime('%H:%M') if o.delivered_at else "Unfulfilled"
                    folium.CircleMarker(
                        location=o.coords, radius=5, color=m_color, fill=True, fill_opacity=0.7,
                        popup=f"<b>Order:</b> {o.id}<br><b>Placed:</b> {o.order_time.strftime('%H:%M')}<br><b>Delivered:</b> {deliv_time}<br><b>Vehicle:</b> Truck_{o.assigned_vehicle}"
                    ).add_to(m)

                TimestampedGeoJson(
                    {"type": "FeatureCollection", "features": features},
                    period="PT2M", transition_time=15, auto_play=True, loop=False, date_options='HH:mm'
                ).add_to(m)

                html_path = "sim_result.html"
                m.save(html_path)

                st.markdown("### 🗺️ Dynamic GPS Replay Context")
                with open(html_path, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=650)

# --- NAVIGATION PIPELINE BUTTONS ---
st.divider()
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])

if st.session_state.current_step > 0:
    if nav_col1.button("⬅️ Back"):
        st.session_state.current_step -= 1
        st.rerun()

if st.session_state.current_step < len(step_names) - 1:
    disabled = False
    if st.session_state.current_step == 0 and 'city_data' not in st.session_state: disabled = True
    if st.session_state.current_step == 1 and 'orders' not in st.session_state: disabled = True

    if nav_col2.button("Next ➡️", disabled=disabled):
        st.session_state.current_step += 1
        st.rerun()
