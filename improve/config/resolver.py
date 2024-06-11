import importlib
import os.path as osp

import hydra
from omegaconf import OmegaConf

import improve


def r_tag_bonus(bonus):
    return "bonus" if bonus else "no-bonus"


def r_toint(val):
    return int(val)


def r_tofloat(val):
    return float(val)

def r_tostring(val):
    return str(val)

def r_home(s):
    return osp.join(osp.expanduser("~"), s)


def r_typeof(class_path):
    """get  a class from a string."""

    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing or instantiating class '{class_path}': {e}")


def r_instantiate(class_path, *args, **kwargs):
    """Instantiate a class from a string."""

    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        return instance

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing or instantiating class '{class_path}': {e}")


def r_ifelse(cond, true_val, false_val):
    return true_val if cond else false_val


def r_weights(path):
    """Load weights from a file."""
    if path is None:
        return None
    return osp.join(improve.WEIGHTS, path)


OmegaConf.register_new_resolver("r_tag_bonus", r_tag_bonus)
OmegaConf.register_new_resolver("r_toint", r_toint)
OmegaConf.register_new_resolver("r_tofloat", r_tofloat)
OmegaConf.register_new_resolver("r_home", r_home)
OmegaConf.register_new_resolver("r_tostring", r_tostring)

OmegaConf.register_new_resolver("r_instantiate", r_instantiate)
OmegaConf.register_new_resolver("r_typeof", r_typeof)
OmegaConf.register_new_resolver("r_ifelse", r_ifelse)

OmegaConf.register_new_resolver("r_weights", r_weights)
