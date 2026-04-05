import math
import random
import time
import folium
from folium.plugins import TimestampedGeoJson
import webbrowser
import os
from datetime import datetime, timedelta


# --- 1. Real-World Distance Calculation ---
def calculate_distance(coord1, coord2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def interpolate_points(coord1, coord2, steps):
    """Linear interpolation for smooth map animation"""
    points = []
    dlat = (coord2[0] - coord1[0]) / steps
    dlon = (coord2[1] - coord1[1]) / steps
    for i in range(steps + 1):
        points.append((coord1[0] + dlat * i, coord1[1] + dlon * i))
    return points


# --- 2. Simulation Logic ---
class Order:
    def __init__(self, order_id, coords, order_time, deadline):
        self.id = order_id
        self.coords = coords
        self.order_time = order_time
        self.deadline = deadline
        self.delivered_at = None


class DeliverySimulation:
    def __init__(self, depot_coords, start_time, end_time, vehicle_speed_kmh=40):
        self.depot = depot_coords
        self.start_time = start_time
        self.end_time = end_time
        self.vehicle_speed_mps = vehicle_speed_kmh * (1000 / 3600)
        self.orders = []
        self.trajectory = []  # List of (timestamp, (lat, lon))
        self.total_distance = 0.0

    def generate_random_orders(self, num_orders, time_limit_hours=2.0):
        """Generates random orders clustered around the city during the day."""
        lat_min, lat_max = 48.10, 48.20
        lon_min, lon_max = 17.00, 17.20

        simulation_duration = (self.end_time - self.start_time).total_seconds()

        for i in range(num_orders):
            lat = random.uniform(lat_min, lat_max)
            lon = random.uniform(lon_min, lon_max)
            # Bias orders to arrive mainly in the first 75% of the day
            order_time_offset = random.uniform(0, simulation_duration * 0.75)
            order_time = self.start_time + timedelta(seconds=order_time_offset)
            deadline = order_time + timedelta(hours=time_limit_hours)

            self.orders.append(Order(f"Order_{i + 1}", (lat, lon), order_time, deadline))

        # Time-series simulations require chronological order
        self.orders.sort(key=lambda o: o.order_time)

    def _travel(self, start_loc, end_loc, start_time):
        """Calculates travel distance, time, and generates animation waypoints."""
        dist = calculate_distance(start_loc, end_loc)
        self.total_distance += dist
        travel_time_sec = dist / self.vehicle_speed_mps
        end_time = start_time + timedelta(seconds=travel_time_sec)

        # Create waypoints every ~30 seconds of travel for smooth Folium animation
        steps = max(1, int(travel_time_sec / 30))
        interpolated = interpolate_points(start_loc, end_loc, steps)

        for i, p in enumerate(interpolated):
            t = start_time + timedelta(seconds=(travel_time_sec / steps) * i)
            self.trajectory.append((t, p))

        return end_time

    def run(self):
        print("\n" + "=" * 50)
        print(f"🚀 RUNNING DYNAMIC SIMULATION ({self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')})")
        print("=" * 50)

        current_time = self.start_time
        current_loc = self.depot
        self.trajectory.append((current_time, current_loc))

        on_time, late = 0, 0

        for order in self.orders:
            # 1. Idle Time Handling (Wait at depot until next order arrives)
            if current_time < order.order_time:
                current_time = order.order_time
                self.trajectory.append((current_time, current_loc))

                # 2. Hub-and-Spoke Requirement: Pick up goods at Depot
            if current_loc != self.depot:
                current_time = self._travel(current_loc, self.depot, current_time)
                current_loc = self.depot

            # Assume 5 mins loading time at depot
            current_time += timedelta(minutes=5)
            self.trajectory.append((current_time, current_loc))

            # 3. Deliver to Customer
            current_time = self._travel(current_loc, order.coords, current_time)
            current_loc = order.coords
            order.delivered_at = current_time

            # 4. SLA Computation
            if order.delivered_at <= order.deadline:
                on_time += 1
                status = "🟢 ON TIME"
            else:
                late += 1
                status = "🔴 LATE"

            print(
                f"[{current_time.strftime('%H:%M')}] Delivered {order.id} | Deadline: {order.deadline.strftime('%H:%M')} | {status}")

            # 5. Return to Depot
            current_time = self._travel(current_loc, self.depot, current_time)
            current_loc = self.depot

        print("\n" + "-" * 50)
        print("🏁 SIMULATION COMPLETE. Generating metrics...")
        print(f"Total Orders Processed : {len(self.orders)}")
        print(f"SLA Success Rate       : {(on_time / len(self.orders)) * 100:.1f}% ({on_time} On Time, {late} Late)")
        print(f"Total Distance Driven  : {self.total_distance / 1000:.2f} km")
        print("-" * 50)

        return {
            "orders": self.orders,
            "trajectory": self.trajectory
        }


# --- 3. Visualization Logic ---
class DynamicVisualizer:
    def __init__(self, depot_coords):
        self.depot_coords = depot_coords

    def generate_map(self, simulation_results, filename="dynamic_route_map.html"):
        # Center map on Bratislava
        m = folium.Map(location=self.depot_coords, zoom_start=12, tiles="cartodbpositron")

        # 1. Plot the Depot
        folium.Marker(
            self.depot_coords, popup="Depot", icon=folium.Icon(color='black', icon='home')
        ).add_to(m)

        # 2. Plot the Customers (Green = On Time, Red = Late)
        for order in simulation_results["orders"]:
            color = "green" if order.delivered_at <= order.deadline else "red"
            popup_html = f"<b>{order.id}</b><br>Ordered: {order.order_time.strftime('%H:%M')}<br>Deadline: {order.deadline.strftime('%H:%M')}<br>Delivered: {order.delivered_at.strftime('%H:%M')}"

            folium.CircleMarker(
                location=order.coords, radius=7, popup=popup_html,
                color=color, fill=True, fill_opacity=0.8
            ).add_to(m)

        # 3. Compile Temporal Data for Animation
        coordinates, times = [], []
        for t, point in simulation_results["trajectory"]:
            coordinates.append([point[1], point[0]])  # GeoJSON strictly requires [Longitude, Latitude]
            times.append(t.strftime('%Y-%m-%dT%H:%M:%S'))

        features = [
            # The fading line path
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "times": times,
                    "style": {"color": "#3388ff", "weight": 3, "opacity": 0.5}
                }
            },
            # The moving vehicle marker
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": coordinates
                },
                "properties": {
                    "times": times,
                    "icon": "circle",
                    "iconstyle": {"fillColor": "blue", "fillOpacity": 1, "stroke": "true", "radius": 8}
                }
            }
        ]

        # 4. Attach the Timestamped Engine
        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="PT1M",  # Step by 1 Minute
            transition_time=50,
            add_last_point=True,
            auto_play=True,
            loop=False,
            max_speed=20,
            loop_button=True,
            date_options='HH:mm',
            time_slider_drag_update=True
        ).add_to(m)

        m.save(filename)
        file_path = os.path.abspath(filename)
        webbrowser.open(f"file://{file_path}")


# --- 4. Execution ---
if __name__ == "__main__":
    # Settings (Current location setup mapping to Bratislava)
    depot_location = (48.1486, 17.1077)
    start_time = datetime(2026, 4, 5, 8, 0, 0)
    end_time = datetime(2026, 4, 5, 18, 0, 0)

    # Init simulation
    sim = DeliverySimulation(depot_location, start_time, end_time, vehicle_speed_kmh=45)

    # Generate N random orders with a x-hour SLA deadline
    sim.generate_random_orders(num_orders=20, time_limit_hours=1.5)

    # Run logic
    results = sim.run()

    # Pass to isolated visualizer
    print("✨ Opening dynamic animation in your default browser...")
    visualizer = DynamicVisualizer(depot_location)
    visualizer.generate_map(results)
