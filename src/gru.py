from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout

from evaluate import evaluate_model


def build_model(seq_length):
    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))
    model.add(GRU(64, return_sequences=True))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model


if __name__ == "__main__":
    evaluate_model(build_model, "GRU")