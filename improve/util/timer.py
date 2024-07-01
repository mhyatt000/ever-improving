import time

import numpy as np


class Timer:
    def __init__(self, name):
        self.name = name
        self.times = []

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        self.elapsed = end_time - self.start_time
        self.times.append(self.elapsed)


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper

