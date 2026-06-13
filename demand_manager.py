import random
from datetime import timedelta
from shapely.geometry import Point
import osmnx as ox  # Ensure osmnx is imported in this file


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
    def generate_gaussian_demand(cls, city_polygon, start_time, num_orders, hourly_weights, tightness):
        """Legacy method: blindly generates points within a polygon."""
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

    @classmethod
    def generate_realistic_demand(cls, city_polygon, graph, start_time, num_orders, hourly_weights, tightness):
        """
        New method: Generates density-weighted points and snaps them to the nearest
        valid road network node, ensuring no deliveries are placed in forests/water.
        """
        centroid = city_polygon.centroid
        center_lat, center_lon = centroid.y, centroid.x
        orders = []
        attempts = 0

        # Step 1: Generate valid raw coordinates using the Gaussian spread
        valid_lons = []
        valid_lats = []

        # Buffer the attempts to ensure we get enough raw points
        max_attempts = num_orders * 10

        while len(valid_lons) < num_orders and attempts < max_attempts:
            lat = random.gauss(center_lat, tightness)
            lon = random.gauss(center_lon, tightness)
            pnt = Point(lon, lat)

            if city_polygon.contains(pnt):
                valid_lons.append(lon)
                valid_lats.append(lat)
            attempts += 1

        # Fallback if the tightness parameter is too restrictive and we didn't hit our target
        if not valid_lons:
            raise ValueError("Could not generate valid points. Check your 'tightness' parameter.")

        # Step 2: Snap all raw coordinates to the nearest road infrastructure in one fast batch operation
        nearest_nodes = ox.distance.nearest_nodes(graph, X=valid_lons, Y=valid_lats)

        # Step 3: Create the final orders based on the actual street coordinates
        for i, node_id in enumerate(nearest_nodes):
            if len(orders) >= num_orders:
                break

            # Extract actual road node coordinates
            actual_lon = graph.nodes[node_id]['x']
            actual_lat = graph.nodes[node_id]['y']

            hour = random.choices(list(hourly_weights.keys()), weights=list(hourly_weights.values()))[0]
            minute = random.randint(0, 59)
            order_time = start_time.replace(hour=hour, minute=minute)
            deadline = order_time + timedelta(hours=2.5)

            orders.append(Order(f"ORD-{len(orders) + 1:03d}", (actual_lat, actual_lon), order_time, deadline))

        orders.sort(key=lambda o: o.order_time)
        return orders
