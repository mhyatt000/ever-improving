from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


@dataclass
class AWAC:
    name: str = "awac"

    actor_optim_kwargs: OptimizerConfig = field(default_factory=OptimizerConfig)
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256)

    state_dependent_std: bool = False
    critic_lr: float = 3e-4
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    discount: float = 0.99

    tau: float = 0.005
    target_update_period: int = 1
    beta: float = 2.0
    num_samples: int = 1
    replay_buffer_size: Optional[int] = None
