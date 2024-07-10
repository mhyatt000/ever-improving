from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from hydra.core.config_store import ConfigStore
from improve.cn.algo.base import Algo
from improve.cn.algo.sac import SAC, OffPolicy
from improve.util.config import default, store


@dataclass
class Optim:
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
