import pickle

def load_model_from_path(model_path):
    with open(model_path, "rb") as input_file:
        model = pickle.load(input_file)
        return model