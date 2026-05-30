import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping


TRAIN_FILE = "processed/train.csv"
TEST_FILE = "processed/test.csv"

DEFAULT_SCATS_IDS = [2000, 3002, 3120, 4043, 4270]

SEQ_LENGTH = 12
EPOCHS = 50
BATCH_SIZE = 32

RESULT_DIR = "testing"
PLOT_DIR = "testing/plots"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)


def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


def prepare_site_data(df, scats_id):
    site_data = df[df["scats_id"] == scats_id].copy()

    if site_data.empty:
        return None

    site_data["datetime"] = pd.to_datetime(site_data["datetime"])

    site_data = (
        site_data
        .groupby("datetime", as_index=False)["traffic_flow"]
        .sum()
        .sort_values("datetime")
    )

    return site_data["traffic_flow"].values.reshape(-1, 1)


def evaluate_model(build_model, model_name, scats_ids=None):
    if scats_ids is None:
        scats_ids = DEFAULT_SCATS_IDS

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    results = []

    for scats_id in scats_ids:
        print(f"\nEvaluating {model_name} on SCATS {scats_id}")

        train_raw = prepare_site_data(train_df, scats_id)
        test_raw = prepare_site_data(test_df, scats_id)

        if train_raw is None or test_raw is None:
            print(f"SCATS {scats_id} not found.")
            continue

        if len(train_raw) <= SEQ_LENGTH or len(test_raw) <= SEQ_LENGTH:
            print(f"SCATS {scats_id} does not have enough data.")
            continue

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw)

        test_input_raw = np.vstack([train_raw[-SEQ_LENGTH:], test_raw])
        test_scaled = scaler.transform(test_input_raw)

        X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
        X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

        model = build_model(SEQ_LENGTH)

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1,
            shuffle=False
        )

        y_pred_scaled = model.predict(X_test, verbose=0)

        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = scaler.inverse_transform(y_pred_scaled)
        y_pred_actual = np.maximum(y_pred_actual, 0)

        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        r2 = r2_score(y_test_actual, y_pred_actual)

        baseline_scaled = X_test[:, -1, 0].reshape(-1, 1)
        baseline_actual = scaler.inverse_transform(baseline_scaled)

        baseline_mae = mean_absolute_error(y_test_actual, baseline_actual)
        baseline_rmse = np.sqrt(mean_squared_error(y_test_actual, baseline_actual))
        baseline_r2 = r2_score(y_test_actual, baseline_actual)

        model_key = model_name.lower()
        plot_file = f"{PLOT_DIR}/{model_key}_scats_{scats_id}.png"

        plt.figure(figsize=(12, 5))
        plt.plot(y_test_actual[:200], label="Actual")
        plt.plot(y_pred_actual[:200], label="Predicted")
        plt.title(f"{model_name} Prediction for SCATS {scats_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Traffic Flow")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()

        results.append({
            "model": model_name,
            "scats_id": scats_id,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": baseline_r2,
            "plot_file": plot_file
        })

    results_df = pd.DataFrame(results)

    output_file = f"{RESULT_DIR}/{model_name.lower()}_results.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nSaved {model_name} results to {output_file}")

    return results_df