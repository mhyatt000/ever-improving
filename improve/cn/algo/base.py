from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from hydra.core.config_store import ConfigStore
from improve.util.config import default, store

# copied from sb3
Schedule = Callable[[float], float]


@dataclass
class Algo:

    learning_rate: float = 3e-4
    batch_size: int = 256

    stats_window_size: int = 10  # was 100 ... for logging only
    tensorboard_log: Optional[str] = None
    seed: int = 0
    device: str = "auto"
    _init_setup_model: bool = True

    support_multi_env: bool = False
    monitor_wrapper: bool = True

    # maybe fix typing later
    # replay_buffer_class: Optional[Type[ReplayBuffer]] = None
    # supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None
    # action_noise: Optional[ActionNoise] = None
    supported_action_spaces: Optional[Tuple] = None
    action_noise: Optional[Any] = None

    replay_buffer_class: Optional[str] = None # "ReplayBuffer" this is not a buffer
    replay_buffer_kwargs: Optional[Dict[str, Any]] = None
    optimize_memory_usage: bool = False

    policy_kwargs: Optional[Dict[str, Any]] = default(
        {
            "share_features_extractor": True,
            "optimizer_kwargs": {
                "betas": (0.999, 0.999),
                "weight_decay": 1e-4,
            },
        }
    )
