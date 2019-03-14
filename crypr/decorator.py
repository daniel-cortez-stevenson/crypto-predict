"""Logging wrappers for functions"""
import logging
import time
from functools import wraps


def my_logger(orig_func):
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info('Running in with args: {}, and kwargs: {}'.format(args, kwargs))
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        logging.info('Ran in: {} sec'.format(time.time() - t1))
        return result
    return wrapper

