"""
PulseRoute Logistics Engine - Algorithmic Core

Local Search (2-Opt) sequencing algorithm
"""

"""
PulseRoute Logistics Engine - Algorithmic Core

Local Search (2-Opt) sequencing algorithm with descriptive logging.
"""

import math
import time


def calculate_haversine_distance(coord1, coord2):
    """Calculates straight-line distance in meters between two GPS coordinates."""
    R = 6371000
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_route_distance(start_location, route):
    """Helper function to calculate the total distance of a given route sequence."""
    if not route:
        return 0.0

    total_dist = calculate_haversine_distance(start_location, route[0].coords)
    for i in range(len(route) - 1):
        total_dist += calculate_haversine_distance(route[i].coords, route[i + 1].coords)
    return total_dist


def greedy_nearest_neighbor(start_location, orders_batch):
    """
    Used to generate a strong initial starting point for the local search.
    """
    route_orders = []
    temp_loc = start_location
    unrouted = list(orders_batch)

    while unrouted:
        next_order = min(unrouted, key=lambda o: calculate_haversine_distance(temp_loc, o.coords))
        route_orders.append(next_order)
        temp_loc = next_order.coords
        unrouted.remove(next_order)

    return route_orders


def local_search_2opt(start_location, orders_batch, route_id="Unknown Route", time_limit_seconds=5.0):
    """
    Implements a 2-Opt local search improvement algorithm with explicit, descriptive logging.
    """
    order_count = len(orders_batch)

    print(f"\n==================================================")
    print(f" STARTING OPTIMIZATION: {route_id}")
    print(f" Total Orders in Batch: {order_count}")
    print(f"==================================================")

    if order_count < 2:
        print(f"[Info] Batch has too few stops ({order_count}). Skipping optimization.")
        print(f" STATUS: Complete | Time: 0.000s\n")
        return orders_batch

    # 1. Generate initial solution
    current_route = greedy_nearest_neighbor(start_location, orders_batch)
    best_distance = calculate_route_distance(start_location, current_route)

    print(f"[Step 1: Baseline] Nearest Neighbor sequence distance: {best_distance:,.2f} meters")
    print(f"[Step 2: Refining] Untangling crossed paths via 2-Opt local search...")

    # 2. Local Search Improvement loop
    start_time = time.time()
    improvement_found = True
    iterations = 0

    while improvement_found:
        improvement_found = False

        for i in range(len(current_route) - 1):
            for j in range(i + 1, len(current_route)):

                # Check time limit
                if time.time() - start_time > time_limit_seconds:
                    print(f"  └── [TIMEOUT] Hit limit of {time_limit_seconds}s. Returning best found so far.")
                    return current_route

                # Reverse the segment between index i and j
                new_route = current_route[:i] + current_route[i:j + 1][::-1] + current_route[j + 1:]
                new_distance = calculate_route_distance(start_location, new_route)

                # If the swap results in a shorter route, accept it
                if new_distance < best_distance - 0.001:
                    saved_distance = best_distance - new_distance
                    current_route = new_route
                    best_distance = new_distance
                    improvement_found = True
                    iterations += 1

                    # Track exact improvement steps
                    print(
                        f"  └── [Fix #{iterations}] Re-ordered segment: Shaved off {saved_distance:.2f} meters (New Total: {best_distance:,.2f} m)")
                    break

            if improvement_found:
                break

    elapsed_time = time.time() - start_time

    # 3. Final Summary Block
    print(f"--------------------------------------------------")
    print(f" SUMMARY FOR: {route_id}")
    print(
        f"  ▪ Initial Distance: {calculate_route_distance(start_location, greedy_nearest_neighbor(start_location, orders_batch)):,.2f} meters")
    print(f"  ▪ Final Distance:   {best_distance:,.2f} meters")
    print(f"  ▪ Total Corections: {iterations} structural bottlenecks resolved")
    print(f"  ▪ Execution Time:   {elapsed_time:.3f} seconds")
    print(f"==================================================\n")

    return current_route


def base_route_sequencer(start_location, orders_batch, route_id="Route_Asset"):
    """
    INTERFACE FUNCTION FOR ROUTING ALGORITHMS.

    Passing a dynamic string to `route_id` is possible here
    (e.g., f"Driver_{index}" or f"Zone_{zone_name}").
    """
    return local_search_2opt(start_location, orders_batch, route_id=route_id, time_limit_seconds=10.0)
