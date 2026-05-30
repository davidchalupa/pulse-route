"""
PulseRoute Logistics Engine - Parameterized Global VRPTW Solver
Builds optimal, capacity-respecting trips (depot returns) with configurable
SLA protection modes: Hard Constraints vs Dynamic Penalty Balancing.
"""

import math
import time
from datetime import timedelta

def calculate_haversine_distance(coord1, coord2):
    R = 6371000
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def local_search_2opt_time_aware(start_location, orders_batch, speed_mps):
    """Refines a trip while penalizing SLA breaks."""
    current_route = orders_batch

    def evaluate_route(route):
        current_time = route[0].order_time
        current_loc = start_location
        dist = 0
        late_count = 0

        for o in route:
            d = calculate_haversine_distance(current_loc, o.coords)
            dist += d
            current_time = max(current_time + timedelta(seconds=d / speed_mps), o.order_time)
            if current_time > o.deadline:
                late_count += 1
            current_time += timedelta(minutes=2)
            current_loc = o.coords
        return dist, late_count

    best_dist, best_late = evaluate_route(current_route)
    improvement_found = True
    start_time = time.time()

    while improvement_found:
        improvement_found = False
        for i in range(len(current_route) - 1):
            for j in range(i + 1, len(current_route)):
                if time.time() - start_time > 1.0:
                    return current_route

                new_route = current_route[:i] + current_route[i:j + 1][::-1] + current_route[j + 1:]
                new_dist, new_late = evaluate_route(new_route)

                # Accept if it improves distance without worsening the SLA
                if new_late < best_late or (new_late == best_late and new_dist < best_dist - 0.001):
                    current_route = new_route
                    best_dist = new_dist
                    best_late = new_late
                    improvement_found = True
                    break
            if improvement_found: break

    return current_route


def global_cvrp_solver(depot_coords, orders, num_vehicles, vehicle_capacity, speed_kmh, sla_mode="hard"):
    """
    Builds discrete 'Trips' (Depot -> Deliveries -> Depot) ensuring vehicles
    never exceed capacity and respect time windows.

    :param sla_mode: "hard" to completely reject late route choices,
                     "dynamic" to use soft cost-penalties.
    """
    speed_mps = speed_kmh * (1000 / 3600)

    # Sort primarily by deadline urgency, then by order time.
    unassigned = sorted(orders, key=lambda o: (o.deadline, o.order_time))

    v_clocks = {i: unassigned[0].order_time for i in range(num_vehicles)}
    v_trips = {i: [] for i in range(num_vehicles)}

    while unassigned:
        v_id = min(range(num_vehicles), key=lambda idx: v_clocks[idx])
        current_clock = v_clocks[v_id]

        best_seed = None
        best_seed_score = float('inf')

        for o in unassigned:
            dist_to_o = calculate_haversine_distance(depot_coords, o.coords)
            travel_time = timedelta(seconds=dist_to_o / speed_mps)
            arrival = max(current_clock + travel_time, o.order_time)

            if sla_mode == "hard" and arrival > o.deadline:
                continue

            wait_time = max(0, (o.order_time - (current_clock + travel_time)).total_seconds())
            time_left = max(0, (o.deadline - arrival).total_seconds())

            # Incorporate time_left. Urgent orders (smaller time_left) score better.
            # 0.5 is a scaling factor to balance seconds with meters.
            score = dist_to_o + (wait_time * 2) + (time_left * 0.5)

            if score < best_seed_score:
                best_seed_score = score
                best_seed = o

        if best_seed is None and unassigned:
            # Fallback still triggers if no valid routes exist, but prioritize urgency here too
            best_seed = min(unassigned, key=lambda o: calculate_haversine_distance(depot_coords, o.coords) + (
                        o.deadline - current_clock).total_seconds())

        trip = [best_seed]
        unassigned.remove(best_seed)

        current_loc = best_seed.coords
        dist_to_seed = calculate_haversine_distance(depot_coords, best_seed.coords)
        trip_clock = max(current_clock + timedelta(seconds=dist_to_seed / speed_mps), best_seed.order_time) + timedelta(
            minutes=2)

        while len(trip) < vehicle_capacity and unassigned:
            best_next = None
            best_next_score = float('inf')

            for o in unassigned:
                dist = calculate_haversine_distance(current_loc, o.coords)
                travel = timedelta(seconds=dist / speed_mps)
                arr = max(trip_clock + travel, o.order_time)
                wait = max(0, (o.order_time - (trip_clock + travel)).total_seconds())

                if wait < 2700:
                    time_left = (o.deadline - arr).total_seconds()

                    if sla_mode == "hard":
                        if arr > o.deadline:
                            continue
                        # Apply urgency weight to the next-order selection
                        score = dist + (wait * 4) + (time_left * 0.5)
                    else:
                        sla_penalty = 100000 if time_left < 0 else 0
                        # Mitigate via software cost penalty factors + urgency
                        score = dist + (wait * 4) + (max(0, time_left) * 0.5) + sla_penalty

                    if score < best_next_score:
                        best_next_score = score
                        best_next = o

            if best_next:
                trip.append(best_next)
                unassigned.remove(best_next)
                dist = calculate_haversine_distance(current_loc, best_next.coords)
                trip_clock = max(trip_clock + timedelta(seconds=dist / speed_mps), best_next.order_time) + timedelta(
                    minutes=2)
                current_loc = best_next.coords
            else:
                break

        optimized_trip = local_search_2opt_time_aware(depot_coords, trip, speed_mps)
        v_trips[v_id].append(optimized_trip)

        sim_clock = current_clock
        sim_loc = depot_coords
        for o in optimized_trip:
            d = calculate_haversine_distance(sim_loc, o.coords)
            sim_clock = max(sim_clock + timedelta(seconds=d / speed_mps), o.order_time) + timedelta(minutes=2)
            sim_loc = o.coords

        d_return = calculate_haversine_distance(sim_loc, depot_coords)
        v_clocks[v_id] = sim_clock + timedelta(seconds=d_return / speed_mps)

    return v_trips
