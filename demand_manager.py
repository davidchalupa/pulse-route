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

    # --- Legacy Methods Kept for Compatibility ---
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
    def generate_gaussian_snapped_demand(cls, city_polygon, graph, start_time, num_orders, hourly_weights, tightness):
        """Previous method: Snaps to any network node including tracks/forest paths."""
        centroid = city_polygon.centroid
        center_lat, center_lon = centroid.y, centroid.x
        orders = []
        valid_lons, valid_lats = [], []
        attempts = 0
        while len(valid_lons) < num_orders and attempts < (num_orders * 10):
            lat = random.gauss(center_lat, tightness)
            lon = random.gauss(center_lon, tightness)
            if city_polygon.contains(Point(lon, lat)):
                valid_lons.append(lon)
                valid_lats.append(lat)
            attempts += 1

        if not valid_lons: return []
        nearest_nodes = ox.distance.nearest_nodes(graph, X=valid_lons, Y=valid_lats)
        for node_id in nearest_nodes:
            if len(orders) >= num_orders: break
            actual_lon = graph.nodes[node_id]['x']
            actual_lat = graph.nodes[node_id]['y']
            hour = random.choices(list(hourly_weights.keys()), weights=list(hourly_weights.values()))[0]
            order_time = start_time.replace(hour=hour, minute=random.randint(0, 59))
            orders.append(Order(f"ORD-{len(orders) + 1:03d}", (actual_lat, actual_lon), order_time,
                                order_time + timedelta(hours=2.5)))
        return orders

    # --- New Ultimate Refined Method ---
    @classmethod
    def generate_realistic_demand(cls, city_polygon, graph, start_time, num_orders, hourly_weights, tightness):
        """
        Refined method: Filters out improbable roads (forest tracks, service paths,
        agricultural links) before snapping to guarantee deliveries land in populated areas.
        """
        # 1. Filter out unwanted road types
        # Common OSM tags for forest roads, fire tracks, and non-residential paths
        excluded_highway_types = {'track', 'service', 'path', 'footway', 'cycleway', 'bridleway', 'unclassified'}

        valid_nodes = set()

        # Iterate through edges to see which nodes connect to valid, probable streets
        for u, v, k, data in graph.edges(keys=True, data=True):
            highway = data.get('highway', '')

            # Handle cases where highway attribute is a list (OSM combined types)
            if isinstance(highway, list):
                is_excluded = any(h in excluded_highway_types for h in highway)
            else:
                is_excluded = highway in excluded_highway_types

            if not is_excluded:
                valid_nodes.add(u)
                valid_nodes.add(v)

        # Fallback: If filtering completely empties the graph (e.g., weird tiny custom map), use all nodes
        if not valid_nodes:
            valid_nodes = set(graph.nodes)

        # Create a subgraph containing only our high-probability delivery nodes
        delivery_subgraph = graph.subgraph(valid_nodes)

        # 2. Generate raw Gaussian points
        centroid = city_polygon.centroid
        center_lat, center_lon = centroid.y, centroid.x
        orders = []
        valid_lons = []
        valid_lats = []
        attempts = 0
        max_attempts = num_orders * 15

        while len(valid_lons) < num_orders and attempts < max_attempts:
            lat = random.gauss(center_lat, tightness)
            lon = random.gauss(center_lon, tightness)
            pnt = Point(lon, lat)

            if city_polygon.contains(pnt):
                valid_lons.append(lon)
                valid_lats.append(lat)
            attempts += 1

        if not valid_lons:
            raise ValueError("Could not generate valid points. Check your 'tightness' parameter or boundary.")

        # 3. Snap strictly to our filtered delivery subgraph
        nearest_nodes = ox.distance.nearest_nodes(delivery_subgraph, X=valid_lons, Y=valid_lats)

        # 4. Construct final orders
        for i, node_id in enumerate(nearest_nodes):
            if len(orders) >= num_orders:
                break

            actual_lon = delivery_subgraph.nodes[node_id]['x']
            actual_lat = delivery_subgraph.nodes[node_id]['y']

            hour = random.choices(list(hourly_weights.keys()), weights=list(hourly_weights.values()))[0]
            minute = random.randint(0, 59)
            order_time = start_time.replace(hour=hour, minute=minute)
            deadline = order_time + timedelta(hours=2.5)

            orders.append(Order(f"ORD-{len(orders) + 1:03d}", (actual_lat, actual_lon), order_time, deadline))

        orders.sort(key=lambda o: o.order_time)
        return orders
