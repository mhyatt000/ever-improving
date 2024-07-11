from improve.util.config import default
from dataclasses import dataclass
from typing import Any, List
from omegaconf import MISSING

from .algo import *
from .buffer import base
from .buffer.base import Buffer
from .env.foundation.base import *

defaults = [
    # Load the config "mysql" from the config group "db"
    {"db": "mysql"}
]


@dataclass
class Config:
    # this is unfortunately verbose due to @dataclass limitations
    defaults: List[Any] = default(defaults)

    # Hydra will populate this field based on the defaults list
    db: Any = MISSING
