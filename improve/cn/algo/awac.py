import os.path as osp
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from improve.cn.algo.base import Algo
from improve.util.config import default, store

from .sac import OffPolicy


@store
@dataclass
class AWAC(OffPolicy):
    name: str = "awac"

    beta: float = 3.3 # was 2.0
    num_samples: int = 3
    gripper_loss_weight: float = 1.0

    ent_coef: str = "0"
    target_entropy: str = "0"

    # offline_steps: int = int(5e5)
    # offline_steps: int = int(5e4) # 50k is 2x from the paper (25k)
    offline_steps: int = int(2.5e4)
    batch_size: int = 1024
    gradient_steps: int = 4
    buffer_size: int = int(1e5)

    log_path: str = "${r_home:improve_logs}"  # head dir of logs
    dataset: Optional[List[str]] = default(["sunny-eon-12"])
    # dataset: Optional[List[str]] = default(["sunny-eon-12", "sleek-shadow-99"])

    learning_starts: int = 0 # immediately after offline

    def __post_init__(self):
        if self.dataset is not None:
            self.dataset = osp.join(self.log_path, self.dataset)
            print(f"dataset: {self.dataset}")
