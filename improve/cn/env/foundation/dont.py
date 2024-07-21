
import os.path as osp
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from improve import names
from improve.util.config import default, store

from .base import FoundationModel

@store
@dataclass
class Dont(FoundationModel):
    name: Optional[str] = None
    ckpt: Optional[str] = None
    task: str = "google_robot_pick_horizontal_coke_can"
