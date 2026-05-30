

from algorithms.astar_travel_time import astar_search
from src.scats_graph import build_travel_time_graph


def main():
    model_type = "lstm"
    origin = 2000
    destination = 3002

    graph = build_travel_time_graph(model_type=model_type)
    path, total_time = astar_search(graph, origin, destination)

    print("\nTBRGS Dummy SCATS Demo Result")
    print("Model:", model_type.upper())
    print("Origin:", origin)
    print("Destination:", destination)
    print("Path:", path)
    print("Estimated travel time:", round(total_time, 2), "minutes")


if __name__ == "__main__":
    main()
