import random
from datetime import timedelta
from shapely.geometry import Point



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
