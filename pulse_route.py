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


# --- 1. Geospatial & Boundary Helpers ---
def get_city_data(city_name):
    """Fetches city boundary and center with better error handling."""
    print(f"🌍 Querying OpenStreetMap for: {city_name}...")

    # Use a unique User-Agent as required by Nominatim's Policy
    headers = {
        'User-Agent': 'PulseRouteSimulation_v2_Contact_YourName'
    }

    # Ensure the city name is URL-encoded properly
    params = {
        'q': city_name,
        'polygon_geojson': 1,
        'format': 'json',
        'limit': 1
    }

    url = "https://nominatim.openstreetmap.org/search"

    try:
        response = requests.get(url, headers=headers, params=params)

        # Check if the server blocked us or had an error
        if response.status_code != 200:
            print(f"❌ Server Error: {response.status_code}")
            print(f"Response content: {response.text[:200]}")  # Show first 200 chars of error
            raise ConnectionError(f"Nominatim API returned status {response.status_code}")

        data = response.json()

        if not data:
            raise ValueError(f"No results found for '{city_name}'. Check spelling or try 'City, Country'.")

        city_info = data[0]
        center_coords = (float(city_info['lat']), float(city_info['lon']))

        if 'geojson' not in city_info:
            raise ValueError(f"The result for {city_name} did not contain boundary (polygon) data.")

        boundary_shape = shape(city_info['geojson'])
        return center_coords, boundary_shape

    except requests.exceptions.JSONDecodeError:
        print("❌ Critical: The server returned HTML instead of JSON. You might be rate-limited.")
        print(f"Server response starts with: {response.text[:100]}")
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
    R = 6371000
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def interpolate_points(coord1, coord2, steps):
    points = []
    dlat = (coord2[0] - coord1[0]) / steps
    dlon = (coord2[1] - coord1[1]) / steps
    for i in range(steps + 1):
        points.append((coord1[0] + dlat * i, coord1[1] + dlon * i))
    return points


# --- 2. Enhanced Demand Generation ---
class DemandManager:
    DEMAND_LEVELS = {"low": 12, "medium": 30, "high": 60}
    HOURLY_WEIGHTS = {8: 0.05, 9: 0.15, 10: 0.25, 11: 0.20, 12: 0.15, 13: 0.10, 14: 0.05, 15: 0.03, 16: 0.02}

    @classmethod
    def generate_realistic_demand(cls, city_polygon, start_time, level="medium", tightness=0.015):
        """
        Generates demand using a Gaussian distribution centered on the city center.
        'tightness' controls the standard deviation (spread).
        """
        num_orders = cls.DEMAND_LEVELS.get(level.lower(), 20)
        centroid = city_polygon.centroid
        center_lat, center_lon = centroid.y, centroid.x

        orders = []
        attempts = 0

        while len(orders) < num_orders and attempts < 1000:
            # Gaussian spread: Draw points from a normal distribution around the center
            lat = random.gauss(center_lat, tightness)
            lon = random.gauss(center_lon, tightness)

            pnt = Point(lon, lat)
            if city_polygon.contains(pnt):
                # Valid point found, now assign a time
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
    def __init__(self, depot_coords, orders, start_time, end_time, vehicle_speed_kmh=45):
        self.depot = depot_coords
        self.orders = orders
        self.start_time = start_time
        self.end_time = end_time
        self.vehicle_speed_mps = vehicle_speed_kmh * (1000 / 3600)
        self.trajectory = []
        self.total_distance = 0.0

    def _travel(self, start_loc, end_loc, start_time):
        dist = calculate_distance(start_loc, end_loc)
        self.total_distance += dist
        travel_time_sec = dist / self.vehicle_speed_mps
        end_time = start_time + timedelta(seconds=travel_time_sec)

        steps = max(1, int(travel_time_sec / 180))  # Interpolate every 3 mins for performance
        interpolated = interpolate_points(start_loc, end_loc, steps)

        for i, p in enumerate(interpolated):
            t = start_time + timedelta(seconds=(travel_time_sec / steps) * i)
            self.trajectory.append((t, p))

        return end_time

    def run(self):
        print("\n" + "=" * 50)
        print(f"🚀 RUNNING DYNAMIC SIMULATION")
        print("=" * 50)

        current_time = self.start_time
        current_loc = self.depot
        self.trajectory.append((current_time, current_loc))
        on_time, late = 0, 0

        for order in self.orders:
            if current_time < order.order_time:
                current_time = order.order_time
                self.trajectory.append((current_time, current_loc))

            if current_loc != self.depot:
                current_time = self._travel(current_loc, self.depot, current_time)
                current_loc = self.depot

            current_time += timedelta(minutes=5)  # Loading time
            self.trajectory.append((current_time, current_loc))

            current_time = self._travel(current_loc, order.coords, current_time)
            current_loc = order.coords
            order.delivered_at = current_time

            if order.delivered_at <= order.deadline:
                on_time += 1
                status = "🟢 ON TIME"
            else:
                late += 1
                status = "🔴 LATE"

            print(
                f"[{current_time.strftime('%H:%M')}] Delivered {order.id} | Deadline: {order.deadline.strftime('%H:%M')} | {status}")

            current_time = self._travel(current_loc, self.depot, current_time)
            current_loc = self.depot

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

    def generate_map(self, simulation_results, boundary_polygon, filename="global_route_map.html"):
        m = folium.Map(location=self.depot_coords, zoom_start=12, tiles="cartodbpositron")

        # Draw the City Boundary to prove the geofencing worked
        folium.GeoJson(
            boundary_polygon,
            style_function=lambda x: {'color': 'gray', 'fillOpacity': 0.1, 'weight': 2}
        ).add_to(m)

        folium.Marker(self.depot_coords, popup="Depot", icon=folium.Icon(color='black', icon='home')).add_to(m)

        for order in simulation_results["orders"]:
            color = "green" if order.delivered_at <= order.deadline else "red"
            folium.CircleMarker(
                location=order.coords, radius=6,
                popup=f"<b>{order.id}</b><br>Ordered: {order.order_time.strftime('%H:%M')}<br>Delivered: {order.delivered_at.strftime('%H:%M')}",
                color=color, fill=True, fill_opacity=0.8
            ).add_to(m)

        coordinates, times = [], []
        for t, point in simulation_results["trajectory"]:
            coordinates.append([point[1], point[0]])
            times.append(t.strftime('%Y-%m-%dT%H:%M:%S'))

        features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {"times": times, "style": {"color": "#3388ff", "weight": 3, "opacity": 0.5}}
            },
            {
                "type": "Feature",
                "geometry": {"type": "MultiPoint", "coordinates": coordinates},
                "properties": {"times": times, "icon": "circle",
                               "iconstyle": {"fillColor": "blue", "fillOpacity": 1, "stroke": "true", "radius": 7}}
            }
        ]

        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="PT1M", transition_time=50, add_last_point=True, auto_play=True, loop=False, max_speed=1,
            date_options='HH:mm'
        ).add_to(m)

        m.save(filename)
        webbrowser.open(f"file://{os.path.abspath(filename)}")


# --- 5. Execution ---
if __name__ == "__main__":
    # 1. Choose ANY city (Format: "City, Country" is safest)
    TARGET_CITY = "Vienna, Austria"
    DEMAND_LEVEL = "low"  # Options: "low", "medium", "high"
    TIGHTNESS = 0.03

    start_time = datetime(2026, 4, 5, 8, 0, 0)
    end_time = datetime(2026, 4, 5, 18, 0, 0)

    # 2. Fetch the geofence
    depot_location, city_polygon = get_city_data(TARGET_CITY)

    # 3. Generate demand strictly within the fence based on a realistic time curve
    orders = DemandManager.generate_realistic_demand(city_polygon, start_time, level=DEMAND_LEVEL, tightness=TIGHTNESS)

    # 4. Run the simulation
    sim = DeliverySimulation(depot_location, orders, start_time, end_time, vehicle_speed_kmh=45)
    results = sim.run()

    # 5. Visualize (Now includes the city boundary line)
    visualizer = DynamicVisualizer(depot_location)
    visualizer.generate_map(results, city_polygon)
