from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from improve.cn.algo.base import Algo
from improve.util.config import default, store

# copied from sb3
Schedule = Callable[[float], float]

# TrainFreq = namedtuple("TrainFreq", ["frequency", "unit"])


@dataclass
class OffPolicy(Algo):

    buffer_size: int = int(1e6)
    learning_starts: int = int(1e4)
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1

    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    sde_support: bool = True

    use_original_space: bool = True
    warmup_zero_action: bool = True


@dataclass
class SAC(OffPolicy):
    name: str = "sac"

    # ent_coef: Union[float | str] = "auto_0.1"
    # target_entropy: Union[float | str] = -7  # "auto"
    ent_coef: str = "auto_0.1"
    target_entropy: str = "-7"  # "auto"

    target_update_interval: int = 1
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False

    use_original_space: bool = False
    warmup_zero_action: bool = False


@dataclass
class ResidualPolicy(Algo):
    name: str = "residual_policy"

    l2_weight: float = 1.0


@store
@dataclass
class RP_SAC(SAC, ResidualPolicy):
    name: str = "rp_sac"
    support_multi_env: bool = True
