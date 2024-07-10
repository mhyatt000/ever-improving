from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from improve.util.config import default, store

LowDimKeys = [
    "agent_qpos-sin",
    "agent_qpos-cos",
    "agent_qvel",
    "agent_partial-action",
    "eef-pose",
]

OracleKeys = [
    "obj-wrt-eef",
]

ImageKeys = [
    "simple-img",
]


class Mode(Enum):
    RGB = "rgb"
    STATE_DICT = "state_dict"


@dataclass
class ObsMode:
    name: str = "base"
    mode: Mode = "rgb"


@dataclass
class Oracle(ObsMode):
    name: str = "oracle"
    obs_keys: List[str] = default(OracleKeys + LowDimKeys)


@dataclass
class OracleCentral(Oracle):
    name: str = "oracle-central"
    obs_keys: List[str] = default(OracleKeys + LowDimKeys + ImageKeys)


class LowDim(ObsMode):
    name: str = "lowdim"
    mode: str = "state_dict"
    obs_keys: None = None


class Image(ObsMode):
    name: str = "image"
    obs_keys: List[str] = default(ImageKeys + LowDimKeys)


class Hybrid(ObsMode):
    name: str = "hybrid"
    obs_keys: List[str] = default(ImageKeys + OracleKeys + LowDimKeys)

    def __init__(self):
        raise NotImplementedError
