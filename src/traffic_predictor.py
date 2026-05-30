import os

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


DATA_FILE = "processed/all_data.csv"
SEQ_LENGTH = 12


MODEL_PATHS = {
    "lstm": {
        "model": "models/lstm/lstm_model.h5",
        "scaler": "models/lstm/lstm_scaler.pkl",
    },
    "gru": {
        "model": "models/gru/gru_model.h5",
        "scaler": "models/gru/gru_scaler.pkl",
    },
    "rnn": {
        "model": "models/rnn/rnn_model.h5",
        "scaler": "models/rnn/rnn_scaler.pkl",
    },
}


def load_scats_series(scats_id):
    """
    Load traffic flow data for one SCATS site from the processed CSV file.

    Args:
        scats_id: SCATS site number, for example 2000.

    Returns:
        numpy.ndarray: traffic flow values shaped as (n, 1).
    """

    df = pd.read_csv(DATA_FILE)
    site_data = df[df["scats_id"] == scats_id].copy()

    if site_data.empty:
        raise ValueError(f"SCATS site {scats_id} not found in dataset.")

    site_data["datetime"] = pd.to_datetime(site_data["datetime"])
    site_data = site_data.sort_values("datetime")

    return site_data["traffic_flow"].values.reshape(-1, 1)


def predict_next_flow(scats_id, model_type="lstm", seq_length=SEQ_LENGTH):
    """
    Predict the next 15-minute traffic flow for a SCATS site.

    Args:
        scats_id: SCATS site number.
        model_type: "lstm", "gru", or "rnn".
        seq_length: number of previous readings used for prediction.

    Returns:
        float: predicted traffic flow for the next 15-minute interval.
    """

    model_type = model_type.lower()

    if model_type not in MODEL_PATHS:
        raise ValueError("model_type must be 'lstm', 'gru', or 'rnn'.")

    model_path = MODEL_PATHS[model_type]["model"]
    scaler_path = MODEL_PATHS[model_type]["scaler"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    traffic_flow = load_scats_series(scats_id)
    traffic_scaled = scaler.transform(traffic_flow)

    last_sequence = traffic_scaled[-seq_length:]
    X = np.array([last_sequence]).reshape(1, seq_length, 1)

    prediction_scaled = model.predict(X, verbose=0)
    prediction_actual = scaler.inverse_transform(prediction_scaled)

    return float(prediction_actual[0][0])


if __name__ == "__main__":
    predicted = predict_next_flow(scats_id=2000, model_type="lstm")
    print("Predicted next 15-min traffic flow:", round(predicted, 2))
