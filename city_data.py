import requests
import os
from shapely.geometry import shape
import osmnx as ox
import pickle


def get_city_data(city_name):
    """Fetches city boundary, default center, and road network with caching."""
    headers = {'User-Agent': 'PulseRouteSimulation_v5_Streamlit'}
    params = {'q': city_name, 'polygon_geojson': 1, 'format': 'json', 'limit': 1}
    url = "https://nominatim.openstreetmap.org/search"

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200 or not response.json():
        raise ValueError(f"Could not find or fetch data for '{city_name}'.")

    data = response.json()[0]

    # Reconcile duplicate caching by tracking the unique OpenStreetMap entity ID
    osm_id = data.get('osm_id', 'unknown')
    osm_type = data.get('osm_type', 'node')
    unique_city_key = f"{osm_type}_{osm_id}"
    canonical_name = data.get('display_name', city_name).replace("/", "-").replace("\\", "-").replace(" ", "_")

    cache_dir = "city_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Locate matching file by prefix to prevent alternative naming schemas from duplicating data
    target_file = None
    for f in os.listdir(cache_dir):
        if f.startswith(f"{unique_city_key}__") and f.endswith(".pkl"):
            target_file = f
            break

    if target_file:
        cache_path = os.path.join(cache_dir, target_file)
    else:
        cache_path = os.path.join(cache_dir, f"{unique_city_key}__{canonical_name}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    center_coords = (float(data['lat']), float(data['lon']))
    boundary_shape = shape(data['geojson'])

    # Download graph for the exact city polygon
    graph = ox.graph_from_polygon(boundary_shape, network_type='drive')

    city_data = (center_coords, boundary_shape, graph)
    with open(cache_path, 'wb') as f:
        pickle.dump(city_data, f)

    return city_data
