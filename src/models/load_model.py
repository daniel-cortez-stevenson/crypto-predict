import pickle
from keras.models import load_model


def load_from_pickle(path):
    with open(path, "rb") as input_file:
        model = pickle.load(input_file)
        return model


def load_from_keras(path):
    model = load_model(path)
    return model