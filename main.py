from __future__ import annotations

import argparse

import pandas as pd

from algorithms.astar_travel_time import find_top_k_routes
from src.scats_graph import build_travel_time_graph


def main(origin=2000, destination=3002, model_type="lstm", k=5):
    model_type = str(model_type).lower()
    origin = int(origin)
    destination = int(destination)
    k = int(k)

    # 1. Check if origin/destination exist in the traffic dataset
    df = pd.read_csv("processed/all_data.csv")
    available_scats_in_dataset = set(df["scats_id"].unique())

    if origin not in available_scats_in_dataset:
        print(f"Origin SCATS {origin} is not in processed/all_data.csv.")
        print("This SCATS site cannot be used because the ML model has no traffic data for it.")
        return [], {}

    if destination not in available_scats_in_dataset:
        print(f"Destination SCATS {destination} is not in processed/all_data.csv.")
        print("This SCATS site cannot be used because the ML model has no traffic data for it.")
        return [], {}

    # 2. Build travel-time graph
    graph = build_travel_time_graph(
        model_type=model_type,
        verbose=False
    )

    # TEMP - remove after checking
    sample_node = next(iter(graph))
    print("graph type:", type(graph))
    print("node type:", type(sample_node))
    print("neighbours type:", type(graph[sample_node]))
    print("sample neighbours:", list(graph[sample_node])[:3])
    

    # 3. Check if origin/destination exist in the generated graph
    if origin not in graph:
        print(f"Origin SCATS {origin} exists in the dataset, but it is not in the generated graph.")
        print("This means graph_builder did not create any edges for this SCATS site.")
        return [], {}

    if destination not in graph:
        print(f"Destination SCATS {destination} exists in the dataset, but it is not in the generated graph.")
        print("This means graph_builder did not create any edges for this SCATS site.")
        return [], {}

    # 4. Find up to k routes using repeated A*
    routes = find_top_k_routes(
        graph=graph,
        origin=origin,
        destination=destination,
        k=k
    )

    print("\nTBRGS Route Result")
    print("Model:", model_type.upper())
    print("Origin:", origin)
    print("Destination:", destination)

    if not routes:
        print("No route found between these SCATS sites.")
        return [], graph

    print("\nTop routes:")
    for index, (path, total_time) in enumerate(routes, start=1):
        print(f"{index}. Path: {path}")
        print(f"   Estimated travel time: {round(total_time, 2)} minutes")

    return routes, graph


def parse_args():
    parser = argparse.ArgumentParser(description="Run the COS30019 TBRGS route calculation.")
    parser.add_argument("--model", dest="model_type", default="lstm", choices=["lstm", "gru", "rnn"], help="Prediction model to use.")
    parser.add_argument("--origin", default=2000, type=int, help="Origin SCATS site number.")
    parser.add_argument("--destination", default=3002, type=int, help="Destination SCATS site number.")
    parser.add_argument("--k", default=5, type=int, help="Maximum number of routes to return.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    routes, _ = main(
        origin=args.origin,
        destination=args.destination,
        model_type=args.model_type,
        k=args.k,
    )