import pandas as pd

from src.traffic_predictor import predict_next_flow
from src.travel_time import haversine_distance, calculate_travel_time


DATA_FILE = "processed/all_data.csv"


# Dummy/demo SCATS connections for testing the integration pipeline.
# Each tuple means: (from_scats, to_scats).
# These are not meant to be the final verified Boroondara road links yet.
SCATS_CONNECTIONS = [
    (2000, 2200),
    (2000, 2820),
    (2200, 3002),
    (2820, 3002),
]


def load_scats_metadata():
    """
    Load unique SCATS site metadata from the processed CSV file.

    Returns:
        dict[int, dict]:
        {
            2000: {
                "location": "...",
                "latitude": -37.8,
                "longitude": 145.0
            }
        }
    """

    df = pd.read_csv(DATA_FILE)

    metadata_df = df[[
        "scats_id",
        "location",
        "latitude",
        "longitude",
    ]].drop_duplicates(subset=["scats_id"])

    metadata = {}

    for _, row in metadata_df.iterrows():
        scats_id = int(row["scats_id"])
        metadata[scats_id] = {
            "location": row["location"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }

    return metadata


def build_coordinate_map():
    """
    Build a coordinate lookup for SCATS sites.

    Returns:
        dict[int, tuple[float, float]]:
        {
            2000: (latitude, longitude)
        }
    """

    metadata = load_scats_metadata()
    return {
        scats_id: (info["latitude"], info["longitude"])
        for scats_id, info in metadata.items()
    }


def build_distance_graph():
    """
    Build a demo graph where edge cost is straight-line distance in kilometres.

    Returns:
        dict[int, list[tuple[int, float]]]:
        {
            2000: [(2200, 1.8), (2820, 2.6)],
            2200: [(3002, 3.5)]
        }
    """

    metadata = load_scats_metadata()
    graph = {}

    for from_scats, to_scats in SCATS_CONNECTIONS:
        graph.setdefault(from_scats, [])
        graph.setdefault(to_scats, [])

        if from_scats not in metadata or to_scats not in metadata:
            print(f"Warning: missing metadata for {from_scats} or {to_scats}")
            continue

        lat1 = metadata[from_scats]["latitude"]
        lon1 = metadata[from_scats]["longitude"]
        lat2 = metadata[to_scats]["latitude"]
        lon2 = metadata[to_scats]["longitude"]

        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
        graph[from_scats].append((to_scats, distance_km))

    return graph


def build_travel_time_graph(model_type="lstm", verbose=True):
    """
    Build a demo graph where edge cost is estimated travel time in minutes.

    Args:
        model_type: "lstm", "gru", or "rnn".
        verbose: if True, prints the predicted flow and edge travel times.

    Returns:
        dict[int, list[tuple[int, float]]]: graph with travel-time edge costs.
    """

    distance_graph = build_distance_graph()
    travel_time_graph = {}

    for from_scats, neighbours in distance_graph.items():
        travel_time_graph[from_scats] = []

        if not neighbours:
            continue

        predicted_flow = predict_next_flow(
            scats_id=from_scats,
            model_type=model_type,
        )

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
    print("=== DEMO DISTANCE GRAPH ===")
    distance_graph = build_distance_graph()
    for node, edges in distance_graph.items():
        print(node, "->", edges)

    print("\n=== DEMO TRAVEL TIME GRAPH ===")
    travel_graph = build_travel_time_graph(model_type="lstm")
    for node, edges in travel_graph.items():
        print(node, "->", edges)
