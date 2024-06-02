from gymnasium.spaces.dict import Dict


def todict(thing):
    """gymnasium.spaces.dict.Dict to dict"""
    if type(thing) is Dict:
        return {k: todict(v) for k, v in thing.spaces.items()}
    else:
        return thing


def apply(d, func):
    """Recursively apply func to items in d."""

    if isinstance(d, Dict):
        return Dict(
            {k: apply(v, func) for k, v in d.spaces.items()}
        )
    elif isinstance(d, dict):
        return {k: apply(v, func) for k, v in d.items()}
    elif isinstance(d, list):
        return [apply(item, func) for item in d]
    else:
        return func(d)


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

