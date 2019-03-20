"""Utility functions"""
import os
import dotenv
import pickle


def get_project_path():
    return os.path.dirname(dotenv.find_dotenv())


def load_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
