from algorithms.astar_travel_time import find_top_k_paths
from src.scats_graph import build_travel_time_graph


def main():
    model_type = "lstm"
    origin = 2000
    destination = 3002

    graph = build_travel_time_graph(model_type=model_type)

    routes = find_top_k_paths(graph, origin, destination, k=5)

    print("\nTBRGS Simplified SCATS Integration Result")
    print("Model:", model_type.upper())
    print("Origin:", origin)
    print("Destination:", destination)

    if not routes:
        print("No route found.")
        return

    print("\nTop routes:")
    for index, (path, total_time) in enumerate(routes, start=1):
        print(f"{index}. Path: {path}")
        print(f"   Estimated travel time: {round(total_time, 2)} minutes")


if __name__ == "__main__":
    main()