import os

import pandas as pd

from src.traffic_predictor import predict_next_flow
from src.travel_time import calculate_travel_time


DATA_FILE = "processed/all_data.csv"
EDGE_FILE = "processed/graph_edges.csv"


def load_scats_metadata():
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

    for node in graph:
        graph[node] = sorted(graph[node], key=lambda edge: edge[0])

    return graph


def build_travel_time_graph(model_type="lstm", verbose=True):
    distance_graph = build_distance_graph(bidirectional=True)

    travel_time_graph = {}
    prediction_cache = {}

    for from_scats, neighbours in distance_graph.items():
        travel_time_graph[from_scats] = []

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