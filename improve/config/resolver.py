import os.path as osp

import hydra
import improve
from omegaconf import OmegaConf


def r_tag_bonus(bonus):
    return "bonus" if bonus else "no-bonus"


def r_toint(val):
    return int(val)


def r_tofloat(val):
    return float(val)


def r_home(s):
    return osp.join(osp.expanduser("~"), s)


OmegaConf.register_new_resolver("r_tag_bonus", r_tag_bonus)
OmegaConf.register_new_resolver("r_toint", r_toint)
OmegaConf.register_new_resolver("r_tofloat", r_tofloat)
OmegaConf.register_new_resolver("r_home", r_home)
