from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense

from evaluate import evaluate_model


def build_model(seq_length):
    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))
    model.add(SimpleRNN(64, activation="tanh"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    return model


if __name__ == "__main__":
    evaluate_model(build_model, "RNN")