import os
import webbrowser
from datetime import datetime

# Shared backend imports
from demand_manager import DemandManager
from delivery_simulation import DeliverySimulation
import routing_engine_nn
import routing_engine_ls_2opt
import routing_engine_cvrp

# Import city data helper from app or shared utils
from pulse_route import get_city_data


def run_benchmark(city_name="Bratislava, Slovakia", num_orders=50, num_vehicles=2):
    print("=" * 80)
    print(f"🚀 PULSEROUTE MULTI-ENGINE BENCHMARK: {city_name}")
    print("=" * 80)

    # 1. Fetch map data
    depot_loc, boundary, graph = get_city_data(city_name)
    start_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)

    # 2. Generate demand
    print(f"📦 Generating {num_orders} simulated orders...")
    weights = {8: 2, 9: 3, 10: 2, 11: 4, 12: 12, 13: 5, 14: 3, 15: 4, 16: 6}
    orders = DemandManager.generate_realistic_demand(
        city_polygon=boundary,
        graph=graph,
        start_time=start_date,
        num_orders=num_orders,
        hourly_weights=weights,
        tightness=0.03
    )

    # 3. Define the full suite of optimization strategies
    strategies = {
        "Local (Nearest Neighbor)": {
            "route_algorithm": routing_engine_nn.base_route_sequencer,
            "optimization_scope": "Local",
            "sla_mode": "hard"
        },
        "Local (2-Opt Search)": {
            "route_algorithm": routing_engine_ls_2opt.base_route_sequencer,
            "optimization_scope": "Local",
            "sla_mode": "hard"
        },
        "Global CVRP (Strict SLA)": {
            "route_algorithm": None,
            "optimization_scope": "Global",
            "sla_mode": "hard"
        },
        "Global CVRP (Dynamic SLA)": {
            "route_algorithm": None,
            "optimization_scope": "Global",
            "sla_mode": "dynamic"
        }
    }

    results_summary = {}
    ideal_bound_km = 0.0

    # 4. Execute Benchmark Loop
    for name, params in strategies.items():
        print(f"\n🏃 Running Strategy: {name}...")

        # Reset order state between runs
        for o in orders:
            o.delivered_at = None
            o.assigned_vehicle = None

        sim_params = {
            "num_vehicles": num_vehicles,
            "vehicle_capacity": 5,
            "max_wait_minutes": 25,
            "vehicle_speed_kmh": 45,
            **params
        }

        sim = DeliverySimulation(depot_loc, orders, graph, **sim_params)
        res = sim.run()

        success_rate = (res['on_time'] / len(orders)) * 100
        dist_km = res['total_distance'] / 1000

        # Captured once since orders and depot location remain constant
        ideal_bound_km = res['theoretical_min_distance'] / 1000
        efficiency_gap = (dist_km / ideal_bound_km) if ideal_bound_km > 0 else 1.0

        results_summary[name] = {
            "SLA Success": f"{success_rate:.1f}%",
            "Actual Dist": f"{dist_km:.2f} km",
            "Overhead Factor": f"{efficiency_gap:.2f}x"
        }

    # 5. Output Tabular Summary with Single Ideal Bound Header
    print("\n" + "=" * 80)
    print("📊 BENCHMARK COMPARISON SUMMARY")
    print(f"🎯 Theoretical Lower Bound (Greedy Ideal): {ideal_bound_km:.2f} km")
    print("=" * 80)
    header = f"{'Strategy Name':<28} | {'SLA Success':<12} | {'Actual Dist':<13} | {'Overhead'}"
    print(header)
    print("-" * 80)

    for strat, metrics in results_summary.items():
        row = (
            f"{strat:<28} | "
            f"{metrics['SLA Success']:<12} | "
            f"{metrics['Actual Dist']:<13} | "
            f"{metrics['Overhead Factor']}"
        )
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark(num_orders=50, num_vehicles=2)
