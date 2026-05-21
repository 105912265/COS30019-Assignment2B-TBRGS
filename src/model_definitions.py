from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU

def get_lstm(sequence_length):
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model


def get_gru(sequence_length):
    model = Sequential()
    model.add(GRU(64, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model