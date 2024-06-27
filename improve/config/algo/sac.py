from dataclasses import dataclass
from typing import Optional, Union, List

from hydra.core.config_store import ConfigStore
from improve import CS


@dataclass
class SACCN:
    name: str = "sac"
    policy_kwargs: dict = {"share_features_extractor": True}

    learning_rate: float = 3e-4
    buffer_size: int = 1000000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: str = None

    replay_buffer_class: str = "ReplayBuffer"
    replay_buffer_kwargs: dict = {}
    optimize_memory_usage: bool = False

    ent_coef: Union[float | str] = "auto_0.1"
    target_entropy: Union[float | str] = "auto"

    target_update_interval: int = 1
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False

    stats_window_size: int = 10  # was 100 ... for logging only
    tensorboard_log: str = None
    seed: int = 0
    device: str = "auto"
    _init_setup_model: bool = True

    use_original_space: bool = False
    warmup_zero_action: bool = False
