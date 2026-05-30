import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

from model_definitions import get_gru

SCATS_ID = 2000
SEQ_LENGTH = 12

# load processed data from data_processor.py
train_df = pd.read_csv('processed/train.csv')
test_df  = pd.read_csv('processed/test.csv')
scaler   = joblib.load('scaler.pkl')

# filter for one SCATS site
train_site = train_df[train_df['scats_id'] == SCATS_ID]['traffic_scaled'].values
test_site  = test_df[test_df['scats_id']  == SCATS_ID]['traffic_scaled'].values

# create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_site, SEQ_LENGTH)
X_test,  y_test  = create_sequences(test_site,  SEQ_LENGTH)

# reshape for GRU input (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

# build model
model = get_gru(SEQ_LENGTH)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# train
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# evaluate
y_pred_scaled = model.predict(X_test)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred_scaled)

mae  = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2   = r2_score(y_test_actual, y_pred_actual)

print("\nGRU Results")
print("SCATS Site:", SCATS_ID)
print("MAE: ",  mae)
print("RMSE:",  rmse)
print("R2:  ",  r2)

# plot
plt.figure(figsize=(12, 5))
plt.plot(y_test_actual[:200], label='Actual')
plt.plot(y_pred_actual[:200], label='Predicted')
plt.title(f'GRU Prediction for SCATS {SCATS_ID}')
plt.xlabel('Time Step')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()

# save model and scaler
import os
os.makedirs('models', exist_ok=True)
model.save('models/gru/gru_model.h5')
joblib.dump(scaler, 'models/gru/gru_scaler.pkl')

print("Model saved as models/gru/gru_model.h5")
print("Current directory:", os.getcwd())
os.makedirs('models', exist_ok=True)
model.save('models/gru/gru_model.h5')
print("Model saved successfully")