import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# LSTM traffic flow prediction model
# Uses the previous 12 traffic readings to predict the next 15-minute traffic flow value
# Trained and tested on one SCATS site from the October 2006 Boroondara dataset

DATA_FILE = "Scats Data October 2006.xls"
SCATS_ID = 2000
SEQ_LENGTH = 12 #use previous 12 traffic readings to predict the next reading

df = pd.read_excel(DATA_FILE, sheet_name="Data", header=1)

volume_cols = [f"V{i:02d}" for i in range(96)]
df[volume_cols] = df[volume_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

site_data = df[df["SCATS Number"] == SCATS_ID]

if site_data.empty:
    raise ValueError("SCATS site not found. Try another SCATS ID.")

daily_data = site_data.groupby("Date")[volume_cols].sum().sort_index()
traffic_flow = daily_data.to_numpy().flatten().reshape(-1, 1)

scaler = MinMaxScaler()
traffic_scaled = scaler.fit_transform(traffic_flow)

def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)

X, y = create_sequences(traffic_scaled, SEQ_LENGTH)

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

model = Sequential()
model.add(Input(shape=(SEQ_LENGTH, 1)))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_pred_scaled = model.predict(X_test)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred_scaled)

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

print("LSTM Results")
print("SCATS Site:", SCATS_ID)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(12, 5))
plt.plot(y_test_actual[:200], label="Actual")
plt.plot(y_pred_actual[:200], label="Predicted")
plt.title(f"LSTM Prediction for SCATS {SCATS_ID}")
plt.xlabel("Time Step")
plt.ylabel("Traffic Flow")
plt.legend()
plt.show()

os.makedirs("models", exist_ok=True)

model.save("models/lstm/lstm_model.h5")
joblib.dump(scaler, "models/lstm/lstm_scaler.pkl")

print("Model saved as lstm_model.h5")
print("Scaler saved as lstm_scaler.pkl")