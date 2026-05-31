import pandas as pd

from algorithms.astar_travel_time import find_top_k_routes
from src.scats_graph import build_travel_time_graph


def main():
    model_type = "lstm"

    # Change these for testing
    origin = 2000
    destination = 3002

    # 1. Check if origin/destination exist in the traffic dataset
    df = pd.read_csv("processed/all_data.csv")
    available_scats_in_dataset = set(df["scats_id"].unique())

    if origin not in available_scats_in_dataset:
        print(f"Origin SCATS {origin} is not in processed/all_data.csv.")
        print("This SCATS site cannot be used because the ML model has no traffic data for it.")
        return

    if destination not in available_scats_in_dataset:
        print(f"Destination SCATS {destination} is not in processed/all_data.csv.")
        print("This SCATS site cannot be used because the ML model has no traffic data for it.")
        return

    # 2. Build travel-time graph
    graph = build_travel_time_graph(
        model_type=model_type,
        verbose=False
    )

    # 3. Check if origin/destination exist in the generated graph
    if origin not in graph:
        print(f"Origin SCATS {origin} exists in the dataset, but it is not in the generated graph.")
        print("This means graph_builder did not create any edges for this SCATS site.")
        return

    if destination not in graph:
        print(f"Destination SCATS {destination} exists in the dataset, but it is not in the generated graph.")
        print("This means graph_builder did not create any edges for this SCATS site.")
        return

    # 4. Find up to 5 routes using repeated A*
    routes = find_top_k_routes(
        graph=graph,
        origin=origin,
        destination=destination,
        k=5
    )

    print("\nTBRGS Route Result")
    print("Model:", model_type.upper())
    print("Origin:", origin)
    print("Destination:", destination)

    if not routes:
        print("No route found between these SCATS sites.")
        return

    for index, (path, total_time) in enumerate(routes, start=1):
        print(f"\nRoute {index}")
        print("Path:", path)
        print("Estimated travel time:", round(total_time, 2), "minutes")


if __name__ == "__main__":
    main()