import math
import random
import time
import requests
import folium
from folium.plugins import TimestampedGeoJson
import webbrowser
import os
from datetime import datetime, timedelta
from shapely.geometry import shape, Point
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import pickle


# --- 1. Geospatial & Boundary Helpers ---
def get_city_data(city_name):
    """
    Fetches city boundary, center, and road network with a local
    (pickle-based) caching mechanism to avoid repeated heavy downloads.
    """
    # 1. Setup cache directory and filename
    cache_dir = "city_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Sanitize name for a valid filename (e.g., "vienna_austria.pkl")
    safe_name = city_name.replace(",", "").replace(" ", "_").lower()
    cache_path = os.path.join(cache_dir, f"{safe_name}.pkl")

    # 2. Try to load from cache
    if os.path.exists(cache_path):
        print(f"📦 Loading cached road network for {city_name}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # 3. Cache Miss: Fetch data normally
    print(f"🌍 Cache miss. Querying OpenStreetMap for: {city_name}...")
    headers = {'User-Agent': 'PulseRouteSimulation_v3_RoadAware'}
    params = {'q': city_name, 'polygon_geojson': 1, 'format': 'json', 'limit': 1}
    url = "https://nominatim.openstreetmap.org/search"

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise ConnectionError(f"Nominatim API returned status {response.status_code}")

        data = response.json()
        if not data:
            raise ValueError(f"No results found for '{city_name}'.")

        city_info = data[0]
        center_coords = (float(city_info['lat']), float(city_info['lon']))
        boundary_shape = shape(city_info['geojson'])

        print(f"🛣️ Downloading road network for {city_name} (this will be cached)...")
        # Download the drivable street network.
        # NOTE: We keep the graph unprojected (Lat/Lon) so snapping works correctly.
        graph = ox.graph_from_polygon(boundary_shape, network_type='drive')

        # 4. Save to cache for next time
        city_data = (center_coords, boundary_shape, graph)
        with open(cache_path, 'wb') as f:
            pickle.dump(city_data, f)

        print(f"💾 Data successfully cached at {cache_path}")
        return city_data

    except Exception as e:
        print(f"❌ Error fetching city data: {e}")
        raise


def generate_points_in_polygon(polygon, num_points):
    """Generates random coordinates strictly inside the given shapely polygon."""
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < num_points:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            # Return as (lat, lon) for Folium/Math
            points.append((pnt.y, pnt.x))
    return points


def calculate_distance(coord1, coord2):
    """Haversine distance (kept for legacy/demand logic)."""
    R = 6371000
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# --- 2. Enhanced Demand Generation ---
class DemandManager:
    DEMAND_LEVELS = {"low": 12, "medium": 30, "high": 60}

    # MODIFICATION: Sharp peak at 12:00 PM, tapering off realistically
    HOURLY_WEIGHTS = {
        8: 0.02, 9: 0.03, 10: 0.05, 11: 0.15,
        12: 0.50,  # 50% of the daily demand happens in the noon hour
        13: 0.15, 14: 0.05, 15: 0.03, 16: 0.02
    }

    @classmethod
    def generate_realistic_demand(cls, city_polygon, start_time, level="medium", tightness=0.015):
        num_orders = cls.DEMAND_LEVELS.get(level.lower(), 20)
        centroid = city_polygon.centroid
        center_lat, center_lon = centroid.y, centroid.x

        orders = []
        attempts = 0

        while len(orders) < num_orders and attempts < 1000:
            lat = random.gauss(center_lat, tightness)
            lon = random.gauss(center_lon, tightness)

            pnt = Point(lon, lat)
            if city_polygon.contains(pnt):
                hour = random.choices(list(cls.HOURLY_WEIGHTS.keys()),
                                      weights=list(cls.HOURLY_WEIGHTS.values()))[0]
                minute = random.randint(0, 59)
                order_time = start_time.replace(hour=hour, minute=minute)
                deadline = order_time + timedelta(hours=2.5)

                orders.append(Order(f"ORD-{len(orders) + 1:03d}", (lat, lon), order_time, deadline))
            attempts += 1

        orders.sort(key=lambda o: o.order_time)
        cls.plot_histogram(orders)
        return orders

    @staticmethod
    def plot_histogram(orders):
        hours = [o.order_time.hour for o in orders]
        plt.figure(figsize=(8, 3))
        plt.hist(hours, bins=range(8, 18), align='left', color='#2ecc71', edgecolor='white')
        plt.title("Daily Demand Profile (Hourly Volume)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Orders")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        print("📊 View the demand curve. Close the window to start the vehicle simulation...")
        plt.show()


# --- 3. Simulation Logic ---
class Order:
    def __init__(self, order_id, coords, order_time, deadline):
        self.id = order_id
        self.coords = coords
        self.order_time = order_time
        self.deadline = deadline
        self.delivered_at = None


class DeliverySimulation:
    # MODIFICATION: Added max_wait_minutes and vehicle_capacity
    def __init__(self, depot_coords, orders, start_time, end_time, graph, vehicle_speed_kmh=45, max_wait_minutes=30,
                 vehicle_capacity=5):
        self.depot = depot_coords
        self.orders = orders
        self.start_time = start_time
        self.end_time = end_time
        self.graph = graph
        self.vehicle_speed_mps = vehicle_speed_kmh * (1000 / 3600)
        self.max_wait_minutes = max_wait_minutes
        self.vehicle_capacity = vehicle_capacity
        self.trajectory = []
        self.total_distance = 0.0

    def _get_road_route(self, start_coords, end_coords):
        """Finds the shortest path on the road network between two points."""
        orig_node = ox.nearest_nodes(self.graph, start_coords[1], start_coords[0])
        dest_node = ox.nearest_nodes(self.graph, end_coords[1], end_coords[0])

        try:
            route = nx.shortest_path(self.graph, orig_node, dest_node, weight='length')
            path_coords = []
            path_length = 0

            nodes_data = self.graph.nodes
            for i in range(len(route)):
                node = route[i]
                lat, lon = nodes_data[node]['y'], nodes_data[node]['x']
                path_coords.append((lat, lon))

                if i > 0:
                    edge_data = self.graph.get_edge_data(route[i - 1], route[i])
                    path_length += edge_data[0]['length']

            return path_coords, path_length
        except nx.NetworkXNoPath:
            return [start_coords, end_coords], calculate_distance(start_coords, end_coords)

    def _travel(self, start_loc, end_loc, start_time):
        """Travels along the road network instead of a straight line."""
        path_points, dist_meters = self._get_road_route(start_loc, end_loc)
        self.total_distance += dist_meters
        travel_time_sec = dist_meters / self.vehicle_speed_mps

        if not path_points or len(path_points) < 2:
            return start_time, 0.0

        time_per_segment = travel_time_sec / (len(path_points) - 1)

        for i, p in enumerate(path_points):
            t = start_time + timedelta(seconds=time_per_segment * i)
            self.trajectory.append((t, p))

        return start_time + timedelta(seconds=travel_time_sec), dist_meters

    def run(self):
        print("\n" + "=" * 50)
        print(f"🚀 RUNNING ROAD-AWARE BATCH SIMULATION")
        print("=" * 50)

        current_time = self.start_time
        current_loc = self.depot
        self.trajectory.append((current_time, current_loc))
        on_time, late = 0, 0

        unassigned_orders = sorted(self.orders, key=lambda o: o.order_time)

        while unassigned_orders:
            if current_time < unassigned_orders[0].order_time:
                current_time = unassigned_orders[0].order_time
                self.trajectory.append((current_time, current_loc))

            batch_start_time = unassigned_orders[0].order_time
            dispatch_time = batch_start_time + timedelta(minutes=self.max_wait_minutes)

            batch = []
            for o in list(unassigned_orders):
                if o.order_time <= dispatch_time and len(batch) < self.vehicle_capacity:
                    batch.append(o)
                elif len(batch) >= self.vehicle_capacity:
                    break

            actual_dispatch_time = max(current_time, batch[-1].order_time)
            if actual_dispatch_time > current_time:
                current_time = actual_dispatch_time
                self.trajectory.append((current_time, current_loc))

            current_time += timedelta(minutes=5)  # Loading time

            for o in batch:
                unassigned_orders.remove(o)

            # Nearest Neighbor Routing
            route_orders = []
            temp_loc = self.depot
            unrouted = list(batch)
            while unrouted:
                next_order = min(unrouted, key=lambda o: calculate_distance(temp_loc, o.coords))
                route_orders.append(next_order)
                temp_loc = next_order.coords
                unrouted.remove(next_order)

            # Leg tracking for verbose logging
            last_loc_name = "Depot"

            # Execute the multi-stop route
            for order in route_orders:
                departure_time = current_time
                current_time, leg_dist = self._travel(current_loc, order.coords, current_time)
                current_loc = order.coords
                order.delivered_at = current_time

                status = "🟢 ON TIME" if order.delivered_at <= order.deadline else "🔴 LATE"
                if order.delivered_at <= order.deadline:
                    on_time += 1
                else:
                    late += 1

                # ENHANCED LOGGING
                print(f"[{current_time.strftime('%H:%M')}] Delivered {order.id} | "
                      f"From: {last_loc_name} (Left {departure_time.strftime('%H:%M')}) | "
                      f"Leg: {leg_dist / 1000:.2f} km | "
                      f"Deadline: {order.deadline.strftime('%H:%M')} | {status}")

                last_loc_name = order.id
                current_time += timedelta(minutes=2)  # Dropoff time

            # Return to depot
            current_time, _ = self._travel(current_loc, self.depot, current_time)
            current_loc = self.depot
            self.trajectory.append((current_time, current_loc))

        print("\n" + "-" * 50)
        print("🏁 SIMULATION COMPLETE.")
        print(f"SLA Success Rate       : {(on_time / len(self.orders)) * 100:.1f}% ({on_time} On Time, {late} Late)")
        print(f"Total Distance Driven  : {self.total_distance / 1000:.2f} km")
        print("-" * 50)

        return {"orders": self.orders, "trajectory": self.trajectory}


# --- 4. Visualization Logic ---
class DynamicVisualizer:
    def __init__(self, depot_coords):
        self.depot_coords = depot_coords

    def generate_map(self, simulation_results, boundary_polygon, filename="road_aware_map.html"):
        m = folium.Map(location=self.depot_coords, zoom_start=13, tiles="cartodbpositron")

        folium.GeoJson(
            boundary_polygon,
            style_function=lambda x: {'color': 'gray', 'fillOpacity': 0.05, 'weight': 1}
        ).add_to(m)

        folium.Marker(self.depot_coords, popup="Depot", icon=folium.Icon(color='black', icon='home')).add_to(m)

        for order in simulation_results["orders"]:
            color = "green" if order.delivered_at <= order.deadline else "red"
            folium.CircleMarker(
                location=order.coords, radius=5,
                popup=f"<b>{order.id}</b><br>Ordered: {order.order_time.strftime('%H:%M')}<br>Delivered: {order.delivered_at.strftime('%H:%M')}",
                color=color, fill=True, fill_opacity=0.7
            ).add_to(m)

        coordinates, times = [], []
        for t, point in simulation_results["trajectory"]:
            coordinates.append([point[1], point[0]])
            times.append(t.strftime('%Y-%m-%dT%H:%M:%S'))

        features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {"times": times, "style": {"color": "#3498db", "weight": 4, "opacity": 0.6}}
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coordinates[0]},
                "properties": {"times": times, "icon": "circle",
                               "iconstyle": {"fillColor": "#2980b9", "fillOpacity": 1, "stroke": "true", "radius": 6}}
            }
        ]

        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="PT1M",
            transition_time=15,
            add_last_point=True,
            auto_play=True,
            loop=False,
            max_speed=20,
            date_options='HH:mm'
        ).add_to(m)

        m.save(filename)
        webbrowser.open(f"file://{os.path.abspath(filename)}")


# --- 5. Execution ---
if __name__ == "__main__":
    TARGET_CITY = "Bratislava, Slovakia"
    DEMAND_LEVEL = "medium"
    TIGHTNESS = 0.03

    start_time = datetime(2026, 4, 5, 8, 0, 0)
    end_time = datetime(2026, 4, 5, 18, 0, 0)

    depot_location, city_polygon, road_graph = get_city_data(TARGET_CITY)

    orders = DemandManager.generate_realistic_demand(city_polygon, start_time, level=DEMAND_LEVEL, tightness=TIGHTNESS)

    sim = DeliverySimulation(
        depot_coords=depot_location,
        orders=orders,
        start_time=start_time,
        end_time=end_time,
        graph=road_graph,
        vehicle_speed_kmh=40,
        # max_wait_minutes=0,     # Simulation with no wait allowed
        max_wait_minutes=25,    # Simulation with potential 25 minute waiting
        vehicle_capacity=5
    )

    results = sim.run()

    visualizer = DynamicVisualizer(depot_location)
    visualizer.generate_map(results, city_polygon)
