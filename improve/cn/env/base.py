from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from omegaconf import MISSING

from improve.util.config import default, store, store_as_head


class FMLoc(Enum):
    ENV = "env"
    CENTRAL = "central"


@store
@dataclass
class Env:
    name: str = "base"
    defaults: List[Any] = default(
        [
            {"foundation": "octo-base"},
            {"obs_mode": "oracle-central"},
        ]
    )

    task: str = "${.foundation.task}"
    bonus: bool = False
    kind: str = "sb3"
    downscale: int = 7
    device: Optional[Any] = None
    obs_keys: List[str] = "${env.obs_mode.obs_keys}"

    goal: dict = default(
        {
            "use": True,
            "key": "simpler-img",
            "cls": "${r_typeof:improve.wrapper.GoalEnvWrapper}",
        }
    )

    residual_scale: int = 1
    scale_strategy: str = "clip"
    action_mask_dims: Optional[List[int]] = None

    use_original_space: bool = False # "${algo.use_original_space}"
    # use_wandb: str = "${job.wandb.use}"

    seed: dict = default(
        {
            "force": False,
            "value": "${job.seed}",
            "seeds": None,
        }
    )

    reward: str = "sparse"
    max_episode_steps: int = 60
    n_envs: int = 16
    no_quarternion: bool = False
    reach: bool = False
    fm_loc: FMLoc = FMLoc.CENTRAL
    
    # record dataset
    record: bool = False
