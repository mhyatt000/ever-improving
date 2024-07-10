from dataclasses import dataclass, field
from typing import List, Optional, Union

from improve.util.config import default, store

from .base import Algo

# from improve import CS


@store
@dataclass
class PPO(Algo):
    name: str = "ppo"

    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    target_kl: Optional[float] = None

    n_steps: int = 2048


# Using the type
# cs.store(name="ppo", node=PPO)
