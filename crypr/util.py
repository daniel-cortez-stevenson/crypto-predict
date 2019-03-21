"""Utility functions"""
from os.path import dirname
from dotenv import find_dotenv
import pickle
from datetime import datetime, timezone
import time
from functools import wraps


def get_project_path() -> str:
    return dirname(find_dotenv())


def load_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def utc_timestamp_now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def utc_timestamp_ymd(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())


def my_logger(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        print('Running in with args: {}, and kwargs: {}'.format(args, kwargs))
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        print('Ran in: {} sec'.format(time.time() - t1))
        return result
    return wrapper

