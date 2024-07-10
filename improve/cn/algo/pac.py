from dataclasses import dataclass, field
from typing import List, Optional, Union

from improve.util.config import default, store

from improve.cn.algo.base import Algo
# from improve import CS


# @store
@dataclass
class PAC(Algo):
    name: str = "pac"
    policy_kwargs: dict = default({"share_features_extractor": True})

    # policy: Union[str, Type[SACPolicy]]
    # env: Union[GymEnv, str]

    tau: float = 5e-3
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: Optional[str] = None

    ent_coef: Union[float | str] = "auto_0.1"
    target_update_interval: int = 1
    target_entropy: Union[str, float] = "auto"
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False


# @store
@dataclass
class Trainer:
    learning_rate: float = 3e-4
    batch_size: int = 256
    device: str = "auto"
    seed: int = 0


# @store(name="base", group="log")
@dataclass
class Logger:
    stats_window_size: int = 10  # was 100
    tensorboard_log: str = None
    _init_setup_model: bool = True
