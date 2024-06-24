import functools
from collections import OrderedDict

import numpy as np
from gymnasium.spaces.dict import Dict


def todict(thing):
    """gymnasium.spaces.dict.Dict to dict"""
    if type(thing) is dict:
        return {k: todict(v) for k, v in thing.items()}
    if type(thing) is list:
        return [todict(v) for v in thing]

    if type(thing) is Dict:
        return {k: todict(v) for k, v in thing.spaces.items()}
    if type(thing) is OrderedDict:
        return {k: todict(v) for k, v in thing.items()}

    else:
        return thing


gym2dict = todict


def concat(arr):
    """Concatenate a list of Dicts."""

    def _concat_helper(x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.concatenate([x, y])
        elif isinstance(x, np.ndarray):
            return np.concatenate([x, np.array([y])])
        elif isinstance(y, np.ndarray):
            return np.concatenate([np.array([x]), y])
        else:
            return np.array([x, y])

    return merge(arr, _concat_helper)


def merge(arr, func):
    """Merge a list of Dicts using func
    the more general version of concat
    """
    return functools.reduce(
        lambda a, b: apply_both(a, b, lambda x, y: func(x, y)),
        arr,
    )


def apply(d, func):
    """Recursively apply func to items in d."""

    if isinstance(d, Dict):
        return Dict({k: apply(v, func) for k, v in d.spaces.items()})
    elif isinstance(d, dict):
        return {k: apply(v, func) for k, v in d.items()}
    elif isinstance(d, list):
        return [apply(item, func) for item in d]
    else:
        return func(d)


def apply_both(a, b, func):
    if isinstance(a, Dict) and isinstance(b, Dict):
        return Dict(
            {
                k1: apply_both((v1, v2), func)
                for (k1, v1), (k2, v2) in zip(a.spaces.items(), b.spaces.items())
            }
        )
    elif isinstance(a, dict) and isinstance(b, dict):
        return {
            k1: apply_both(v1, v2, func)
            for (k1, v1), (k2, v2) in zip(a.items(), b.items())
        }
    elif isinstance(a, list) and isinstance(b, list):
        return [apply_both(a, b, func) for a, b in zip(a, b)]
    else:
        return func(a, b)


def flatten(d, delim="_"):
    """flattens a dict. the opposite of dict_nest"""

    def _flatten(subd, parentk=""):
        items = {}
        for k, v in subd.items():
            new = parentk + delim + k if parentk else k
            if isinstance(v, dict):
                items.update(_flatten(v, new))
            else:
                items[new] = v
        return items

    return _flatten(d)


def nest(d, delim="/"):
    result = {}

    for key, value in d.items():
        parts = key.split(delim)
        current_level = result
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[-1]] = value
    return result
