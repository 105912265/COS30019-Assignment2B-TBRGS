import math
import os
import re
from collections import defaultdict, deque

import pandas as pd


DATA_FILE = "processed/all_data.csv"
OUTPUT_FILE = "processed/graph_edges.csv"


ROAD_SUFFIXES = (
    "RD|ROAD|ST|STREET|HWY|HIGHWAY|FWY|FREEWAY|AVE|AVENUE|"
    "PDE|PARADE|DR|DRIVE|CT|COURT|CRES|CRESCENT|LANE|LN"
)

'''
this file creates a reasonable graph by:
        1. Reading unique SCATS sites from processed/all_data.csv
        2. Extracting road names from each site's location text
        3. Connecting neighbouring SCATS sites that share the same road name
        4. Checking whether the graph is disconnected
        5. Adding connector edges between disconnected components if needed
        6. Saving the final edge list to processed/graph_edges.csv

'''

#calculate real-world distance between two latitude/longitude points
def haversine_distance(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    diff_lat = lat2 - lat1
    diff_lon = lon2 - lon1

    a = (
        math.sin(diff_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(diff_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c

#standardise road names
def clean_road_name(text):
    text = str(text).upper()
    text = text.replace(".", "")
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def extract_road_names(location):
    text = str(location).upper()
    text = text.replace(".", "")
    text = text.replace("-", "_")
    text = text.replace("/", "|")
    text = text.replace("&", "|")
    text = text.replace(",", "|")

    text = re.sub(r"\b(N|S|E|W|NE|NW|SE|SW)\s+OF\b", "|", text)
    text = re.sub(r"\bOF\b", "|", text)
    text = re.sub(r"\b(AND|AT|CNR|CORNER)\b", "|", text)

    roads = set()

    for part in text.split("|"):
        part = clean_road_name(part)

        match = re.search(rf"([A-Z0-9_]+_(?:{ROAD_SUFFIXES}))\b", part)

        if match:
            roads.add(clean_road_name(match.group(1)))

    return roads

#load one row per scats side from all_data.csv
def load_unique_sites():
    df = pd.read_csv(DATA_FILE)

    required_columns = {"scats_id", "location", "latitude", "longitude"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {DATA_FILE}: {missing}")

    sites = df[
        ["scats_id", "location", "latitude", "longitude"]
    ].drop_duplicates(subset=["scats_id"]).copy()

    sites["scats_id"] = sites["scats_id"].astype(int)

    sites = sites[
        (sites["latitude"] < -36)
        & (sites["latitude"] > -39)
        & (sites["longitude"] > 143)
        & (sites["longitude"] < 146)
    ].copy()

    sites["roads"] = sites["location"].apply(extract_road_names)

    return sites


def make_site_lookup(sites):
    site_lookup = {}

    for _, row in sites.iterrows():
        site_lookup[int(row["scats_id"])] = {
            "location": row["location"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "roads": row["roads"],
        }

    return site_lookup

#build graph edges using shared road naames
def build_road_edges(site_lookup, max_distance_km=2.5):
    road_groups = defaultdict(list)

    for scats_id, info in site_lookup.items():
        for road in info["roads"]:
            road_groups[road].append(scats_id)

    edge_lookup = {}

    for road, scats_ids in road_groups.items():
        if len(scats_ids) < 2:
            continue

        lat_values = [site_lookup[s]["latitude"] for s in scats_ids]
        lon_values = [site_lookup[s]["longitude"] for s in scats_ids]

        lat_range = max(lat_values) - min(lat_values)
        lon_range = max(lon_values) - min(lon_values)

        if lat_range >= lon_range:
            ordered_sites = sorted(scats_ids, key=lambda s: site_lookup[s]["latitude"])
        else:
            ordered_sites = sorted(scats_ids, key=lambda s: site_lookup[s]["longitude"])

        for from_scats, to_scats in zip(ordered_sites, ordered_sites[1:]):
            distance_km = haversine_distance(
                site_lookup[from_scats]["latitude"],
                site_lookup[from_scats]["longitude"],
                site_lookup[to_scats]["latitude"],
                site_lookup[to_scats]["longitude"],
            )

            if distance_km > max_distance_km:
                continue

            edge_key = tuple(sorted((from_scats, to_scats)))

            if edge_key not in edge_lookup:
                edge_lookup[edge_key] = {
                    "from_scats": edge_key[0],
                    "to_scats": edge_key[1],
                    "road": road,
                    "distance_km": round(distance_km, 4),
                    "edge_type": "road_name_match",
                }

    return edge_lookup

#build edge list into a simple adjacency graph
def build_graph_from_edges(edge_lookup, all_nodes):
    graph = {node: [] for node in all_nodes}

    for edge in edge_lookup.values():
        a = int(edge["from_scats"])
        b = int(edge["to_scats"])

        graph[a].append(b)
        graph[b].append(a)

    return graph

#find connected components in  the graph
#if graph has more than one component, it maens graph is split into separate islands and some routes may fail ebtween SCATS
def find_components(edge_lookup, all_nodes):
    graph = build_graph_from_edges(edge_lookup, all_nodes)

    visited = set()
    components = []

    for node in all_nodes:
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbour in graph[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        components.append(sorted(component))

    return components

#creates connector edges when the graph is disconnected
def closest_pair_between_components(component_a, component_b, site_lookup):
    best_pair = None
    best_distance = float("inf")

    for site_a in component_a:
        for site_b in component_b:
            distance_km = haversine_distance(
                site_lookup[site_a]["latitude"],
                site_lookup[site_a]["longitude"],
                site_lookup[site_b]["latitude"],
                site_lookup[site_b]["longitude"],
            )

            if distance_km < best_distance:
                best_distance = distance_km
                best_pair = (site_a, site_b)

    return best_pair, best_distance

#ensure that the final graph is connected
def connect_components(edge_lookup, site_lookup):
    """
    If the generated graph has disconnected islands, connect them using the
    closest pair of SCATS sites between components.

    These edges are labelled as connector_edge so we know they were inferred.
    """
    all_nodes = sorted(site_lookup.keys())
    components = find_components(edge_lookup, all_nodes)

    print(f"Connected components before fixing: {len(components)}")
    for i, component in enumerate(components, start=1):
        print(f"Component {i}: {len(component)} nodes -> {component}")

    while len(components) > 1:
        best_connection = None
        best_distance = float("inf")
        best_component_pair = None

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                pair, distance = closest_pair_between_components(
                    components[i],
                    components[j],
                    site_lookup,
                )

                if distance < best_distance:
                    best_distance = distance
                    best_connection = pair
                    best_component_pair = (i, j)

        from_scats, to_scats = best_connection
        edge_key = tuple(sorted((from_scats, to_scats)))

        print(
            f"Adding connector edge: {edge_key[0]} <-> {edge_key[1]} "
            f"distance = {best_distance:.4f} km"
        )

        edge_lookup[edge_key] = {
            "from_scats": edge_key[0],
            "to_scats": edge_key[1],
            "road": "CONNECTOR",
            "distance_km": round(best_distance, 4),
            "edge_type": "connector_edge",
        }

        components = find_components(edge_lookup, all_nodes)

    print(f"Connected components after fixing: {len(components)}")

    return edge_lookup


def build_edges():
    sites = load_unique_sites()
    site_lookup = make_site_lookup(sites)

    edge_lookup = build_road_edges(
        site_lookup=site_lookup,
        max_distance_km=2.5,
    )

    print(f"Road-name edges created: {len(edge_lookup)}")

    edge_lookup = connect_components(edge_lookup, site_lookup)

    edges = list(edge_lookup.values())

    return edges


def save_edges():
    os.makedirs("processed", exist_ok=True)

    edges = build_edges()

    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values(["edge_type", "from_scats", "to_scats"])

    edges_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFinal edge count: {len(edges_df)}")
    print(f"Saved to {OUTPUT_FILE}")
    print(edges_df.to_string(index=False))


if __name__ == "__main__":
    save_edges()