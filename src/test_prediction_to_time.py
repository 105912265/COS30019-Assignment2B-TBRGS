from src.traffic_predictor import predict_next_flow
from src.travel_time import calculate_travel_time


def main():
    scats_id = 2000
    distance_km = 2.0
    model_type = "lstm"

    predicted_flow = predict_next_flow(
        scats_id=scats_id,
        model_type=model_type
    )

    travel_time = calculate_travel_time(
        distance_km=distance_km,
        predicted_flow_15min=predicted_flow
    )

    print("Prediction to Travel Time Test")
    print("Model:", model_type.upper())
    print("SCATS Site:", scats_id)
    print("Predicted 15-min flow:", round(predicted_flow, 2))
    print("Distance:", distance_km, "km")
    print("Estimated travel time:", round(travel_time, 2), "minutes")


if __name__ == "__main__":
    main()