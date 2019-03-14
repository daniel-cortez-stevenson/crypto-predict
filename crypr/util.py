"""Utility functions"""
import os
import dotenv
import pickle


def get_project_path():
    project_path = os.path.dirname(dotenv.find_dotenv())
    return project_path


def load_from_pickle(path):
    with open(path, "rb") as input_file:
        model = pickle.load(input_file)
        return model