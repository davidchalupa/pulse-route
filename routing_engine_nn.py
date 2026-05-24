"""
PulseRoute Logistics Engine - Algorithmic Core

Baseline nearest-neighbor sequencing algorithm
"""

import math
from datetime import timedelta


def calculate_haversine_distance(coord1, coord2):
    """Calculates straight-line distance in meters between two GPS coordinates."""
    R = 6371000
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def base_route_sequencer(start_location, orders_batch):
    """
    INTERFACE FUNCTION FOR ROUTING ALGORITHMS.

    To substitute with a different algorithm (e.g., Genetic, OR-Tools, TSP Solver):
    Maintain this exact input and output signature.

    Args:
        start_location (tuple): (latitude, longitude) of starting position (Depot or current vehicle location)
        orders_batch (list): List of Order objects that need to be sequenced.

    Returns:
        list: Reordered/sequenced list of Order objects.
    """
    return greedy_nearest_neighbor(start_location, orders_batch)


def greedy_nearest_neighbor(start_location, orders_batch):
    """
    Executes a standard Greedy Nearest Neighbor sequence strategy.
    Highly sub-optimal but fast. Space for optimization here.
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
