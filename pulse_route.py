import math
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


def calculate_distance(coord1, coord2):
    R = 6371000
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


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
                 max_wait_minutes=30, vehicle_capacity=5):
        self.depot = depot_coords
        self.orders = orders
        self.graph = graph
        self.vehicle_speed_mps = vehicle_speed_kmh * (1000 / 3600)
        self.max_wait_minutes = max_wait_minutes
        self.vehicle_capacity = vehicle_capacity

        # Initialize fleet
        self.vehicles = [{"id": i, "loc": depot_coords, "time": orders[0].order_time if orders else datetime.now(),
                          "trajectory": [], "distance": 0.0} for i in range(num_vehicles)]

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
            return [start_coords, end_coords], calculate_distance(start_coords, end_coords)

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
        unassigned_orders = sorted(self.orders, key=lambda o: o.order_time)
        on_time = 0

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

            # Nearest Neighbor Routing
            route_orders = []
            temp_loc = v["loc"]
            unrouted = list(batch)
            while unrouted:
                next_order = min(unrouted, key=lambda o: calculate_distance(temp_loc, o.coords))
                route_orders.append(next_order)
                temp_loc = next_order.coords
                unrouted.remove(next_order)

            # Execute route
            for order in route_orders:
                self._travel(v, order.coords)
                order.delivered_at = v["time"]
                order.assigned_vehicle = v["id"]
                if order.delivered_at <= order.deadline:
                    on_time += 1
                v["time"] += timedelta(minutes=2)  # Dropoff time

            # Return to depot
            self._travel(v, self.depot)

        total_distance = sum(v["distance"] for v in self.vehicles)
        return {"orders": self.orders, "vehicles": self.vehicles, "on_time": on_time, "total_distance": total_distance}


# --- 4. Stepper Logic & UI ---

# Initialize Step in Session State
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

step_names = ["📍 Location", "📊 Demand", "⚙️ Fleet", "🏁 Simulate"]

# Progress Bar & Headers
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


# --- STEP 3: FLEET ---
elif st.session_state.current_step == 2:
    st.header("Fleet Configuration")
    if 'orders' not in st.session_state:
        st.warning("Please generate demand first.")
    else:
        c1, c2 = st.columns(2)
        num_v = c1.number_input("Vehicles", min_value=1, max_value=10, value=2)
        cap = c1.number_input("Capacity", min_value=1, max_value=20, value=5)
        wait = c2.slider("Max Wait (mins)", 0, 60, 25, help="How long a package waits at the depot to build a batch.")
        spd = c2.slider("Speed (km/h)", 20, 80, 45)

        st.session_state['sim_params'] = {
            "num_vehicles": num_v, "vehicle_capacity": cap,
            "max_wait_minutes": wait, "vehicle_speed_kmh": spd
        }
        st.success("Configuration saved! You are ready to run the simulation.")


# --- STEP 4: SIMULATE ---
elif st.session_state.current_step == 3:
    st.header("Run Simulation & View Results")
    if 'sim_params' not in st.session_state:
        st.warning("Please configure fleet parameters first.")
    else:
        if st.button("🚀 Start Simulation", type="primary"):
            depot_loc, boundary, graph = st.session_state['city_data']
            orders = st.session_state['orders']
            params = st.session_state['sim_params']

            with st.spinner(f"Simulating routing for {params['num_vehicles']} vehicle(s)..."):
                # Reset order delivery status in case of re-runs
                for o in orders: o.delivered_at = None

                sim = DeliverySimulation(depot_loc, orders, graph, **params)
                results = sim.run()

                # --- Metrics ---
                st.subheader("Simulation Results")
                c1, c2, c3 = st.columns(3)
                success_rate = (results['on_time'] / len(orders)) * 100
                c1.metric("SLA Success Rate", f"{success_rate:.1f}%",
                          f"{results['on_time']}/{len(orders)} On Time",
                          delta_color="normal" if success_rate > 90 else "inverse")
                c2.metric("Total Distance Driven", f"{results['total_distance'] / 1000:.2f} km")
                c3.metric("Fleet Used", f"{params['num_vehicles']} Vehicles")

                # --- Map Generation ---
                m = folium.Map(location=depot_loc, zoom_start=13, tiles="cartodbpositron")
                folium.GeoJson(boundary, style_function=lambda x: {'color': 'gray', 'fillOpacity': 0.05}).add_to(m)
                folium.Marker(depot_loc, icon=folium.Icon(color='black', icon='home')).add_to(m)

                # Colors for different vehicles
                colors = ['#3498db', '#e74c3c', '#9b59b6', '#f1c40f', '#e67e22', '#2ecc71']
                features = []

                for v in results["vehicles"]:
                    coords, times = [], []
                    for t, point in v["trajectory"]:
                        coords.append([point[1], point[0]])
                        times.append(t.strftime('%Y-%m-%dT%H:%M:%S'))

                    if coords:
                        color = colors[v["id"] % len(colors)]
                        # Trajectory Line
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {"times": times, "style": {"color": color, "weight": 4, "opacity": 0.7}}
                        })
                        # Moving Vehicle Point
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": coords[0]},
                            "properties": {"times": times, "icon": "circle",
                                           "iconstyle": {"fillColor": color, "fillOpacity": 1, "stroke": "true",
                                                         "radius": 7}}
                        })

                for o in results["orders"]:
                    m_color = "green" if o.delivered_at <= o.deadline else "red"
                    folium.CircleMarker(
                        location=o.coords, radius=5, color=m_color, fill=True, fill_opacity=0.7,
                        popup=f"Order {o.id}<br>Delivered: {o.delivered_at.strftime('%H:%M')}"
                    ).add_to(m)

                TimestampedGeoJson(
                    {"type": "FeatureCollection", "features": features},
                    period="PT2M", transition_time=15, auto_play=True, loop=False, date_options='HH:mm'
                ).add_to(m)

                html_path = "sim_result.html"
                m.save(html_path)

                st.markdown("### 🗺️ GPS Replay")
                with open(html_path, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=650)

# --- NAVIGATION BUTTONS ---
st.divider()
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])

if st.session_state.current_step > 0:
    if nav_col1.button("⬅️ Back"):
        st.session_state.current_step -= 1
        st.rerun()

if st.session_state.current_step < len(step_names) - 1:
    # Disable "Next" if critical data is missing
    disabled = False
    if st.session_state.current_step == 0 and 'city_data' not in st.session_state: disabled = True
    if st.session_state.current_step == 1 and 'orders' not in st.session_state: disabled = True

    if nav_col2.button("Next ➡️", disabled=disabled):
        st.session_state.current_step += 1
        st.rerun()
