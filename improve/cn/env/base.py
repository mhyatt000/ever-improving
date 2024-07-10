from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from improve.util.config import default, store


@dataclass
class Env:
    name: str = "base"
    defaults: List[Any] = default(
        [
            {"foundation": "octo-base"},
            {"obs_mode": "image"},
        ]
    )

    task: str = "${.foundation.task}"
    bonus: bool = False
    kind: str = "sb3"
    downscale: int = 7
    device: Optional[Any] = None
    obs_keys: str = "${env.obs_mode.obs_keys}"

    goal: dict = {
        "use": True,
        "key": "simpler-img",
        "cls": "${r_typeof:improve.wrapper.GoalEnvWrapper}",
    }

    residual_scale: int = 1
    scale_strategy: str = "clip"
    action_mask_dims: Optional[Any] = None
    use_original_space: str = "${algo.use_original_space}"
    use_wandb: str = "${job.wandb.use}"

    seed: dict = {
        "force": False,
        "value": "${job.seed}",
        "seeds": None,
    }

    reward: str = "sparse"
    max_episode_steps: int = 60
    n_envs: int = 16
    no_quarternion: bool = False
    reach: bool = False
    fm_loc: str = "env"
