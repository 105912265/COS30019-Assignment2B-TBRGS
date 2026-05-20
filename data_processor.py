import pandas as pd
import numpy as np
try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("Warning: scikit-learn not installed. Install it with: pip install scikit-learn")
    MinMaxScaler = None
import pickle
import os

def load_raw_data(filepath):
    """
    Load raw Excel data from the Data sheet
    Skips the Notes header rows by using header=1
    """
    df = pd.read_excel(filepath, engine='xlrd', sheet_name='Data', header=1)
    print(f"Loaded {len(df)} rows, {df['SCATS Number'].nunique()} SCATS sites")
    return df


def clean_data(df):
    """
    Keep only useful columns and drop internal VicRoads codes
    """
    volume_cols = [f'V{i:02d}' for i in range(96)]

    useful_cols = ['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date'] + volume_cols
    df = df[useful_cols].copy()

    # rename for clarity
    df = df.rename(columns={
        'SCATS Number': 'scats_id',
        'Location':     'location',
        'NB_LATITUDE':  'latitude',
        'NB_LONGITUDE': 'longitude',
        'Date':         'date'
    })

    print(f"Kept {len(df.columns)} columns: id, location, lat, long, date + 96 volume readings")
    return df


def reshape_to_timeseries(df):
    """
    Step 3: Melt 96 volume columns into individual rows
    Before: 1 row = 1 site, 1 day, 96 readings
    After:  1 row = 1 site, 1 timestamp, 1 reading
    """
    volume_cols = [f'V{i:02d}' for i in range(96)]

    df_melted = df.melt(
        id_vars=['scats_id', 'location', 'latitude', 'longitude', 'date'],
        value_vars=volume_cols,
        var_name='interval',
        value_name='traffic_flow'
    )

    # V00 = 0 mins, V01 = 15 mins, V02 = 30 mins etc
    df_melted['interval_num'] = df_melted['interval'].str[1:].astype(int)

    # build datetime from date + interval
    df_melted['datetime'] = pd.to_datetime(df_melted['date']) + \
                            pd.to_timedelta(df_melted['interval_num'] * 15, unit='min')

    # drop helper columns
    df_melted = df_melted.drop(columns=['date', 'interval', 'interval_num'])

    # sort by site then time
    df_melted = df_melted.sort_values(['scats_id', 'datetime']).reset_index(drop=True)

    print(f"Reshaped to {len(df_melted)} rows (40 sites x 31 days x 96 intervals)")
    return df_melted


def normalize_data(df):
    """
    Normalize traffic_flow to 0-1 range for ML training
    Saves the scaler so we can inverse transform predictions later
    """
    scaler = MinMaxScaler()
    df['traffic_scaled'] = scaler.fit_transform(df[['traffic_flow']])

    # save scaler for later use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Normalized traffic flow: min={df['traffic_flow'].min()}, max={df['traffic_flow'].max()}")
    print(f"Scaled range: {df['traffic_scaled'].min():.2f} to {df['traffic_scaled'].max():.2f}")
    return df, scaler


def train_test_split(df, split_ratio=0.8):
    """
    Split into training and testing sets
    Uses time-based split (not random) to avoid data leakage
    First 80% of dates = train, last 20% = test
    """
    # split by date not by row to avoid leakage
    dates = sorted(df['datetime'].unique())
    split_idx = int(len(dates) * split_ratio)
    split_date = dates[split_idx]

    train = df[df['datetime'] < split_date].copy()
    test  = df[df['datetime'] >= split_date].copy()

    print(f"Train: {len(train)} rows (up to {split_date})")
    print(f"Test:  {len(test)} rows (from {split_date})")
    return train, test


def create_sequences(data, scats_id, sequence_length=96):
    """
    Create input/output sequences for LSTM/GRU
    Each input = 96 time steps (1 full day) of traffic flow
    Each output = next time step prediction
    """
    site_data = data[data['scats_id'] == scats_id]['traffic_scaled'].values

    X, y = [], []
    for i in range(len(site_data) - sequence_length):
        X.append(site_data[i:i + sequence_length])
        y.append(site_data[i + sequence_length])

    return np.array(X), np.array(y)


def process(filepath):
    print("=== Data Processing ===\n")

    print("Step 1: Loading raw data...")
    df = load_raw_data(filepath)

    print("\nStep 2: Cleaning data...")
    df = clean_data(df)

    print("\nStep 3: Reshaping to time series...")
    df = reshape_to_timeseries(df)

    print("\nStep 4: Normalizing...")
    df, scaler = normalize_data(df)

    print("\nStep 5: Train/test split...")
    train, test = train_test_split(df)

    # save processed data
    os.makedirs('processed', exist_ok=True)
    df.to_csv('processed/all_data.csv', index=False)
    train.to_csv('processed/train.csv', index=False)
    test.to_csv('processed/test.csv', index=False)

    print("\n=== Done ===")
    print("Saved: processed/all_data.csv")
    print("Saved: processed/train.csv")
    print("Saved: processed/test.csv")
    print("Saved: scaler.pkl")

    return df, train, test, scaler


if __name__ == "__main__":
    df, train, test, scaler = process('Scats Data October 2006.xls')
    print("\nSample output:")
    print(df.head(10).to_string())
