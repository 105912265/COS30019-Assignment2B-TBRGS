import os

import pandas as pd

from src.traffic_predictor import predict_next_flow
from src.travel_time import calculate_travel_time


DATA_FILE = "processed/all_data.csv"
EDGE_FILE = "processed/graph_edges.csv"


def load_scats_metadata():
    """
    Load one metadata record per SCATS site.

    This includes:
    - SCATS ID
    - location name
    - latitude
    - longitude

    This function is useful if the GUI or route output needs to display
    intersection names beside SCATS IDs.
    """

    df = pd.read_csv(DATA_FILE)

    metadata_df = df[
        ["scats_id", "location", "latitude", "longitude"]
    ].drop_duplicates(subset=["scats_id"])

    metadata = {}

    for _, row in metadata_df.iterrows():
        scats_id = int(row["scats_id"])

        metadata[scats_id] = {
            "location": row["location"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }

    return metadata


def load_generated_edges():
    """
    Load the generated SCATS edges from processed/graph_edges.csv.

    graph_edges.csv is created by running:
        py -m src.graph_builder

    Each row represents a connection between two SCATS sites.
    """

    if not os.path.exists(EDGE_FILE):
        raise FileNotFoundError(
            f"{EDGE_FILE} was not found. Run this first: py -m src.graph_builder"
        )

    edges_df = pd.read_csv(EDGE_FILE)

    required_columns = {"from_scats", "to_scats", "distance_km"}
    missing = required_columns - set(edges_df.columns)

    if missing:
        raise ValueError(f"Missing columns in {EDGE_FILE}: {missing}")

    return edges_df


def build_distance_graph(bidirectional=True):
    """
    Build a graph where edge costs are distances in kilometres.

    Output format:
        {
            2000: [(3682, 1.65), (3685, 0.34)],
            3682: [(2000, 1.65), (3126, 1.03)]
        }

    This distance graph is later converted into a travel-time graph.
    """

    edges_df = load_generated_edges()

    graph = {}

    for _, row in edges_df.iterrows():
        from_scats = int(row["from_scats"])
        to_scats = int(row["to_scats"])
        distance_km = float(row["distance_km"])

        graph.setdefault(from_scats, [])
        graph.setdefault(to_scats, [])

        graph[from_scats].append((to_scats, distance_km))

        if bidirectional:
            graph[to_scats].append((from_scats, distance_km))

    # Sort neighbours so output is stable and easier to test/debug.
    for node in graph:
        graph[node] = sorted(graph[node], key=lambda edge: edge[0])

    return graph


def build_travel_time_graph(model_type="lstm", verbose=True):
    """
    Convert the distance graph into a travel-time graph.

    For each edge:
        1. Predict traffic flow for the starting SCATS site.
        2. Use predicted flow + distance to estimate travel time.
        3. Store travel time as the edge cost.

    Output format:
        {
            2000: [(3682, 2.15), (3685, 0.84)],
            3682: [(2000, 2.15), (3126, 1.53)]
        }
    """

    distance_graph = build_distance_graph(bidirectional=True)

    travel_time_graph = {}
    prediction_cache = {}

    for from_scats, neighbours in distance_graph.items():
        travel_time_graph[from_scats] = []

        # Cache predictions so each SCATS site is predicted only once.
        if from_scats not in prediction_cache:
            prediction_cache[from_scats] = predict_next_flow(
                scats_id=from_scats,
                model_type=model_type,
            )

        predicted_flow = prediction_cache[from_scats]

        if verbose:
            print(
                f"Predicted flow for SCATS {from_scats}: "
                f"{predicted_flow:.2f} vehicles/15min"
            )

        for to_scats, distance_km in neighbours:
            travel_time = calculate_travel_time(
                distance_km=distance_km,
                predicted_flow_15min=predicted_flow,
            )

            if verbose:
                print(
                    f"Edge {from_scats} -> {to_scats}: "
                    f"distance = {distance_km:.2f} km, "
                    f"travel time = {travel_time:.2f} min"
                )

            travel_time_graph[from_scats].append((to_scats, travel_time))

    return travel_time_graph


if __name__ == "__main__":
    print("=== GENERATED DISTANCE GRAPH ===")
    distance_graph = build_distance_graph()

    for node, edges in distance_graph.items():
        print(node, "->", edges)