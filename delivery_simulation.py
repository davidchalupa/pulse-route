from datetime import datetime, timedelta
import osmnx as ox
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from math import radians, cos, sin, asin, sqrt

import routing_engine_cvrp  # Core Engine for Global Optimization


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
        dlat = dlat = lat2 - lat1
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
            global_trips = routing_engine_cvrp.global_cvrp_solver(
                self.depot, self.orders, len(self.vehicles), self.vehicle_capacity,
                self.vehicle_speed_kmh, sla_mode=self.sla_mode
            )

            for v in self.vehicles:
                assigned_trips = global_trips.get(v["id"], [])
                if not assigned_trips:
                    continue

                if v["time"] < assigned_trips[0][0].order_time:
                    v["time"] = assigned_trips[0][0].order_time

                for trip in assigned_trips:
                    if not trip:
                        continue

                    v["trajectory"].append((v["time"], v["loc"]))
                    v["time"] += timedelta(minutes=5)

                    for order in trip:
                        if v["time"] < order.order_time:
                            v["time"] = order.order_time
                            v["trajectory"].append((v["time"], v["loc"]))

                        self._travel(v, order.coords)
                        order.delivered_at = v["time"]
                        order.assigned_vehicle = v["id"]
                        if order.delivered_at <= order.deadline:
                            on_time += 1
                        v["time"] += timedelta(minutes=2)

                    self._travel(v, self.depot)

        # ==========================================================
        # STRATEGY B: LOCAL FIXED BATCHES (Original Sequential Constraint)
        # ==========================================================
        else:
            unassigned_orders = sorted(self.orders, key=lambda o: o.order_time)

            while unassigned_orders:
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

                v["time"] = max(v["time"], batch[-1].order_time) + timedelta(minutes=5)

                for o in batch:
                    unassigned_orders.remove(o)

                try:
                    route_orders = self.route_algorithm(v["loc"], batch, route_id=f"Vehicle_{v['id']}_Batch")
                except TypeError:
                    route_orders = self.route_algorithm(v["loc"], batch)

                for order in route_orders:
                    if v["time"] < order.order_time:
                        v["time"] = order.order_time
                    self._travel(v, order.coords)
                    order.delivered_at = v["time"]
                    order.assigned_vehicle = v["id"]
                    if order.delivered_at <= order.deadline:
                        on_time += 1
                    v["time"] += timedelta(minutes=2)

                self._travel(v, self.depot)

        total_distance = sum(v["distance"] for v in self.vehicles)
        return {
            "orders": self.orders,
            "vehicles": self.vehicles,
            "on_time": on_time,
            "total_distance": total_distance,
            "theoretical_min_distance": theoretical_min_distance
        }
