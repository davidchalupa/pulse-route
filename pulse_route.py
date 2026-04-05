import math
import time
import folium
import webbrowser
import os


# --- 1. Real-World Distance Calculation ---
def calculate_distance(coord1, coord2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# --- 2. The Nearest Neighbor Algorithm ---
def get_optimized_route(start_location, unvisited_locations):
    route = []
    current_loc = start_location
    unvisited = unvisited_locations.copy()
    while unvisited:
        next_node = min(unvisited, key=lambda node: calculate_distance(current_loc['coords'], node['coords']))
        route.append(next_node)
        current_loc = next_node
        unvisited.remove(next_node)
    return route


# --- 3. Visualization Helper ---
def create_map(depot, initial_route, dynamic_route, visited_stop):
    m = folium.Map(location=[48.1486, 17.1077], zoom_start=12, tiles="cartodbpositron")

    # Plot Depot
    folium.Marker(depot['coords'], popup="Depot", icon=folium.Icon(color='black', icon='home')).add_to(m)

    # 1. Plot the "Legacy" route (Initial plan) in Light Gray/Dashed
    path_points = [depot['coords']] + [s['coords'] for s in initial_route]
    folium.PolyLine(path_points, color="gray", weight=2, opacity=0.5, dash_array='10', tooltip="Initial Plan").add_to(m)

    # 2. Plot the Actual Pulse-Route (The path actually taken after re-optimization)
    actual_path = [visited_stop['coords']] + [s['coords'] for s in dynamic_route]
    folium.PolyLine(actual_path, color="blue", weight=5, opacity=0.8, tooltip="Dynamic Re-route").add_to(m)

    # Add Markers for all customers
    all_stops = initial_route + dynamic_route
    seen_ids = set()
    for stop in all_stops:
        if stop['id'] not in seen_ids:
            color = 'red' if 'URGENT' in stop['id'] else 'blue'
            folium.CircleMarker(
                location=stop['coords'],
                radius=8,
                popup=stop['id'],
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
            seen_ids.add(stop['id'])

    filename = "route_map.html"
    m.save(filename)
    file_path = os.path.abspath(filename)
    webbrowser.open(f"file://{file_path}")


# --- 4. Simulation Setup ---
depot = {'id': 'Depot (Old Town)', 'coords': (48.1486, 17.1077)}
unvisited_customers = [
    {'id': 'Cust_A (Petržalka)', 'coords': (48.1250, 17.1080)},
    {'id': 'Cust_B (Ružinov)', 'coords': (48.1560, 17.1500)},
    {'id': 'Cust_C (Nové Mesto)', 'coords': (48.1680, 17.1350)},
    {'id': 'Cust_D (Karlova Ves)', 'coords': (48.1550, 17.0600)}
]
urgent_customer = {'id': 'URGENT_SURGE (Devín)', 'coords': (48.1740, 16.9800)}


# --- 5. The Simulation Loop ---
def run_simulation():
    print("\n" + "=" * 50)
    print("🚀 INITIALIZING PULSE-ROUTE ENGINE")
    print("=" * 50)
    time.sleep(1)

    # Step 1: Initial Calculation
    current_vehicle_location = depot
    initial_planned_route = get_optimized_route(current_vehicle_location, unvisited_customers)

    print("\n📍 INITIAL ROUTE CALCULATED:")
    for step, stop in enumerate(initial_planned_route):
        print(f"  Step {step + 1}: {stop['id']}")

    print("\n🚚 Vehicle departing depot...")
    time.sleep(2)

    # Step 2: Visit the first stop
    next_stop = initial_planned_route[0]
    unvisited_customers.remove(next_stop)
    current_vehicle_location = next_stop
    print(f"✅ ARRIVED: Delivered to {current_vehicle_location['id']}.")
    time.sleep(2)

    # Step 3: Dynamic Event
    print("\n" + "!" * 50)
    print("🚨 INCOMING DYNAMIC EVENT: NEW URGENT ORDER RECEIVED")
    print("!" * 50)
    time.sleep(1)
    print(f"Adding {urgent_customer['id']} to the active queue...")
    unvisited_customers.append(urgent_customer)
    time.sleep(1)

    # Step 4: Re-optimization
    print("\n⚙️  RE-OPTIMIZING ROUTE FROM CURRENT POSITION...")
    time.sleep(1.5)
    new_planned_route = get_optimized_route(current_vehicle_location, unvisited_customers)

    print("\n📍 NEW DYNAMIC ROUTE:")
    for step, stop in enumerate(new_planned_route):
        print(f"  Step {step + 1}: {stop['id']}")

    # Step 5: Finish deliveries
    for stop in new_planned_route:
        time.sleep(1.5)
        print(f"✅ ARRIVED: Delivered to {stop['id']}.")

    print("\n🏁 ALL DELIVERIES COMPLETE. Generating map...")
    create_map(depot, initial_planned_route, new_planned_route, current_vehicle_location)
    print("✨ Map opened in browser.")


if __name__ == "__main__":
    run_simulation()
