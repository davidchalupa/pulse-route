import requests
import folium
from folium.plugins import TimestampedGeoJson
import os
from datetime import datetime
from shapely.geometry import shape, Point
import matplotlib.pyplot as plt
import osmnx as ox
import pickle
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from demand_manager import DemandManager
from delivery_simulation import DeliverySimulation

# Import original and new routing engines
import routing_engine_nn
import routing_engine_ls_2opt
#import routing_engine_cvrp  # Core Engine for Global Optimization

# --- App Configuration & Styling ---
st.set_page_config(page_title="PulseRoute Simulator", page_icon="🚚", layout="wide")
st.title("🚚 PulseRoute Logistics Simulator")
st.markdown("Model demand, configure fleets, and visualize dynamic routing.")


# --- 1. Geospatial & Boundary Helpers ---
def get_cached_cities():
    """Scans the cache directory and extracts human-readable city names."""
    cache_dir = "city_cache"
    if not os.path.exists(cache_dir):
        return {}
    cached = {}
    for f in os.listdir(cache_dir):
        if "__" in f and f.endswith(".pkl"):
            parts = f.split("__", 1)
            if len(parts) == 2:
                display_name = parts[1].rsplit(".pkl", 1)[0].replace("-", "/").replace("_", " ")
                cached[display_name] = f
    return cached


@st.cache_resource(show_spinner=False)
def get_city_data(city_name):
    """Fetches city boundary, default center, and road network with caching."""
    headers = {'User-Agent': 'PulseRouteSimulation_v5_Streamlit'}
    params = {'q': city_name, 'polygon_geojson': 1, 'format': 'json', 'limit': 1}
    url = "https://nominatim.openstreetmap.org/search"

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200 or not response.json():
        raise ValueError(f"Could not find or fetch data for '{city_name}'.")

    data = response.json()[0]

    # Reconcile duplicate caching by tracking the unique OpenStreetMap entity ID
    osm_id = data.get('osm_id', 'unknown')
    osm_type = data.get('osm_type', 'node')
    unique_city_key = f"{osm_type}_{osm_id}"
    canonical_name = data.get('display_name', city_name).replace("/", "-").replace("\\", "-").replace(" ", "_")

    cache_dir = "city_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Locate matching file by prefix to prevent alternative naming schemas from duplicating data
    target_file = None
    for f in os.listdir(cache_dir):
        if f.startswith(f"{unique_city_key}__") and f.endswith(".pkl"):
            target_file = f
            break

    if target_file:
        cache_path = os.path.join(cache_dir, target_file)
    else:
        cache_path = os.path.join(cache_dir, f"{unique_city_key}__{canonical_name}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    center_coords = (float(data['lat']), float(data['lon']))
    boundary_shape = shape(data['geojson'])

    # Download graph for the exact city polygon
    graph = ox.graph_from_polygon(boundary_shape, network_type='drive')

    city_data = (center_coords, boundary_shape, graph)
    with open(cache_path, 'wb') as f:
        pickle.dump(city_data, f)

    return city_data


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

# --- STEP 1: LOCATION (UPDATED WITH NORMALIZED CONVENTION) ---
if st.session_state.current_step == 0:
    st.header("Select Operating City")
    st.markdown(
        "Search for a city to load its exact boundaries. Once loaded, **click anywhere on the map** to place your central depot.")

    city_source = st.radio("Choose City Initialization Method:",
                           ["🔍 Search Online (Nominatim API)", "💾 Load from Local Cache"], horizontal=True)

    if city_source == "🔍 Search Online (Nominatim API)":
        city_query = st.text_input("Type City Name (e.g., Prague, Paris, Bratislava)", value="Bratislava, Slovakia")

        if st.button("Search for Matching Cities"):
            if city_query:
                with st.spinner("Searching for location suggestions..."):
                    try:
                        headers = {'User-Agent': 'PulseRouteSimulation_v5_Streamlit'}
                        # Enforce addressdetails execution to extract clean keys
                        params = {'q': city_query, 'format': 'json', 'limit': 10, 'addressdetails': 1}
                        url = "https://nominatim.openstreetmap.org/search"
                        res = requests.get(url, headers=headers, params=params)
                        if res.status_code == 200 and res.json():
                            suggestions = []
                            for item in res.json():
                                addr = item.get('address', {})
                                # Fall back up through standard administrative layers if 'city' is missing
                                city = addr.get('city') or addr.get('town') or addr.get('village') or addr.get(
                                    'municipality')
                                if not city:
                                    city = item.get('display_name', '').split(',')[0].strip()

                                state = addr.get('state')
                                country = addr.get('country')

                                if country:
                                    # Formulate normalized 'City, State, Country' configuration for US entries
                                    if country.lower() in ['united states', 'usa',
                                                           'united states of america'] and state:
                                        clean_name = f"{city}, {state}, {country}"
                                    else:
                                        clean_name = f"{city}, {country}"

                                    if clean_name not in suggestions:
                                        suggestions.append(clean_name)

                            st.session_state['city_suggestions'] = suggestions
                            st.session_state['search_performed'] = True
                        else:
                            st.warning("No suggestions found. Try adjusting your search query.")
                            st.session_state['city_suggestions'] = []
                            st.session_state['search_performed'] = False
                    except Exception as e:
                        st.error(f"Suggestion lookup failed: {e}")
                        st.session_state['city_suggestions'] = []
                        st.session_state['search_performed'] = False

        search_performed = st.session_state.get('search_performed', False)
        suggestions = st.session_state.get('city_suggestions', [])

        city_input = st.selectbox(
            "Confirm City Selection from Results List:",
            options=suggestions if suggestions else ["Please trigger a successful city search first..."],
            disabled=not search_performed
        )

        if st.button("Fetch Map Data", disabled=not search_performed):
            with st.spinner("Fetching boundary and mapping road network (this may take a minute on first run)..."):
                try:
                    st.session_state['city_data'] = get_city_data(city_input)
                    st.success(f"Loaded {city_input} successfully! You can now click the map to move the depot.")
                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        cached_dict = get_cached_cities()
        if not cached_dict:
            st.info("ℹ️ No cached cities found in your local directory yet. Try searching online first!")
        else:
            selected_cached = st.selectbox("Select from already cached cities:", options=list(cached_dict.keys()))
            if st.button("Load Cached Map Data"):
                with st.spinner("Loading cached network graph structure..."):
                    try:
                        cache_path = os.path.join("city_cache", cached_dict[selected_cached])
                        with open(cache_path, 'rb') as f:
                            st.session_state['city_data'] = pickle.load(f)
                        st.success(f"Loaded {selected_cached} successfully from local cache!")
                    except Exception as e:
                        st.error(f"Error loading cached file: {e}")

    if 'city_data' in st.session_state:
        depot_loc, boundary, graph = st.session_state['city_data']

        m = folium.Map(location=depot_loc, zoom_start=12)
        folium.GeoJson(boundary, style_function=lambda x: {'color': 'blue', 'fillOpacity': 0.1}).add_to(m)
        folium.Marker(depot_loc, popup="Depot (Click map to move)", icon=folium.Icon(color='red', icon='home')).add_to(
            m)

        map_data = st_folium(m, height=450, use_container_width=True, key="city_map")

        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            clicked_point = Point(lon, lat)

            if boundary.contains(clicked_point):
                if (lat, lon) != depot_loc:
                    st.session_state['city_data'] = ((lat, lon), boundary, graph)
                    st.rerun()
            else:
                st.warning("⚠️ Please click inside the blue city boundary to place the depot.")


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
            _, boundary, graph = st.session_state['city_data']
            st.session_state['orders'] = DemandManager.generate_realistic_demand(
                city_polygon=boundary,
                graph=graph,
                start_time=start_date,
                num_orders=num_orders,
                hourly_weights=weights,
                tightness=tightness,
            )
            st.success(f"Generated {num_orders} orders successfully!")

        if 'orders' in st.session_state:
            st.markdown("### 📍 Generated Demand Visualization")
            depot_loc, boundary, _ = st.session_state['city_data']

            preview_map = folium.Map(location=depot_loc, zoom_start=12, tiles="cartodbpositron")
            folium.GeoJson(boundary, style_function=lambda x: {'color': 'gray', 'fillOpacity': 0.05}).add_to(
                preview_map)
            folium.Marker(depot_loc, icon=folium.Icon(color='black', icon='home'), popup="Central Depot").add_to(
                preview_map)

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
        st.markdown("### 🎛️ Optimization Scope Selection")

        opt_scope = st.selectbox(
            "Select System Optimization Strategy:",
            options=["Local (Fixed Sequential Batches)", "Global (Dynamic Fleet Assignment)"],
            help="Choose how orders are clustered and dispatched to your fleet."
        )

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
                index=0
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

        summary_placeholder = st.empty()
        with summary_placeholder.container():
            if 'sim_history' in st.session_state and st.session_state['sim_history']:
                with st.expander("📊 Strategy Performance Comparison Summary", expanded=True):
                    st.table(st.session_state['sim_history'])

        if st.button("🚀 Start Simulation", type="primary"):
            depot_loc, boundary, graph = st.session_state['city_data']
            orders = st.session_state['orders']

            params = st.session_state['sim_params'].copy()
            params.update({
                "route_algorithm": chosen_algorithm,
                "optimization_scope": opt_scope,
                "sla_mode": sla_constraint_type
            })

            with st.spinner(f"Simulating routing via {params['optimization_scope']}..."):
                for o in orders:
                    o.delivered_at = None

                sim = DeliverySimulation(depot_loc, orders, graph, **params)
                results = sim.run()

                success_rate = (results['on_time'] / len(orders)) * 100
                actual_distance_km = results['total_distance'] / 1000
                lower_bound_km = results['theoretical_min_distance'] / 1000
                efficiency_gap = (results['total_distance'] / results['theoretical_min_distance']) if results[
                                                                                                          'theoretical_min_distance'] > 0 else 1

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

                with summary_placeholder.container():
                    with st.expander("📊 Collapsible Strategy Performance Comparison Summary", expanded=True):
                        st.table(st.session_state['sim_history'])

                st.subheader("Simulation Performance Analysis")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("SLA Success Rate", f"{success_rate:.1f}%",
                          f"{results['on_time']}/{len(orders)} On Time",
                          delta_color="normal" if success_rate > 90 else "inverse")
                c2.metric("Actual Distance", f"{actual_distance_km:.2f} km")
                c3.metric("Theoretical Min (Greedy Ideal)", f"{lower_bound_km:.2f} km",
                          f"{efficiency_gap:.1f}x Multiplier", delta_color="off")
                c4.metric("Active Fleet", f"{params['num_vehicles']} Vehicles Used")

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
