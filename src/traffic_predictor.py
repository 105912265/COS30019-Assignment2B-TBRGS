import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


DATA_FILE = "Scats Data October 2006.xls"
SEQ_LENGTH = 12


MODEL_PATHS = {
    "lstm": {
        "model": "models/lstm/lstm_model.h5",
        "scaler": "models/lstm/lstm_scaler.pkl"
    },
    "gru": {
        "model": "models/gru/gru_model.h5",
        "scaler": "models/gru/gru_scaler.pkl"
    },
    "rnn": {
        "model": "models/rnn/rnn_model.h5",
        "scaler": "models/rnn/rnn_scaler.pkl"
    }
}


def load_scats_series(scats_id):
    """
    Load traffic flow data for one SCATS site from the Excel dataset.
    Returns a long time-series of traffic flow values.
    """

    df = pd.read_excel(DATA_FILE, sheet_name="Data", header=1)

    volume_cols = [f"V{i:02d}" for i in range(96)]
    df[volume_cols] = df[volume_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    site_data = df[df["SCATS Number"] == scats_id]

    if site_data.empty:
        raise ValueError(f"SCATS site {scats_id} not found in dataset.")

    daily_data = site_data.groupby("Date")[volume_cols].sum().sort_index()

    traffic_flow = daily_data.to_numpy().flatten().reshape(-1, 1)

    return traffic_flow


def predict_next_flow(scats_id, model_type="lstm", seq_length=SEQ_LENGTH):
    """
    Predict the next 15-minute traffic flow for a SCATS site.
    Returns:
        predicted traffic flow for the next 15-minute interval
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

    X = np.array([last_sequence])
    X = X.reshape((X.shape[0], X.shape[1], 1))

    prediction_scaled = model.predict(X, verbose=0)

    prediction_actual = scaler.inverse_transform(prediction_scaled)

    return float(prediction_actual[0][0])


if __name__ == "__main__":
    predicted = predict_next_flow(scats_id=2000, model_type="lstm")

    print("Predicted next 15-min traffic flow:", round(predicted, 2))