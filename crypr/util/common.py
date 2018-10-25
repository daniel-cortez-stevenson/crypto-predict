import os, dotenv


def get_project_path():
    project_path = os.path.dirname(dotenv.find_dotenv())
    return project_path