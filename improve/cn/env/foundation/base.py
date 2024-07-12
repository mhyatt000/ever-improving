import os.path as osp
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from improve import names
from improve.util.config import default, store


class Strategy(Enum):
    CLIP = "clip"
    DYNAMIC = "dynamic"
    NONE = None


@dataclass
class FoundationModel:
    name: str = None
    ckpt: Optional[str] = None
    task: str = None

    noact: list = default([-1, -2, -3, -4])  # no action dimensions
    strategy: Strategy = Strategy.CLIP  # residual scaling strategy
    residual_scale: float = 1.0  # residual scaling factor

    batch_size: int = 8  # number of parallel environments

    def __post_init__(self):

        # follow simpler.OctoInference
        self.policy = self.name
        if self.policy in ["octo-base", "octo-small"]:
            if self.ckpt in [None, "None"] or "rt_1_x" in self.ckpt:
                self.ckpt = self.policy

        # follow simpler.OctoInference
        # build policy
        if "google_robot" in self.task:
            self.policy_setup = "google_robot"
        elif "widowx" in self.task:
            self.policy_setup = "widowx_bridge"
        else:
            raise NotImplementedError()


@store
@dataclass
class OctoS(FoundationModel):
    name: str = "octo-small"
    ckpt: Optional[str] = None
    task: str = "widowx_put_eggplant_in_basket"

    cached: bool = False

@store
@dataclass
class OctoB(FoundationModel):
    name: str = "octo-base"
    ckpt: Optional[str] = None
    task: str = "widowx_put_eggplant_in_basket"


@dataclass
class RT1Model:

    num_image_tokens: int = 81
    num_action_tokens: int = 11
    layer_size: int = 256
    vocab_size: int = 512

    # Use token learner to reduce tokens per image to 81.
    use_token_learner: bool = True
    # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
    world_vector_range: tuple = (-2.0, 2.0)


@store
@dataclass
class RTX(FoundationModel):
    name: str = "rtx"
    task: str = "google_robot_pick_horizontal_coke_can"

    model: RT1Model = default(RT1Model())
    # checkpoint_path: str = "rt_1_x_jax/b321733791_75882326_000900000"
    ckpt: str = osp.join(names.WEIGHTS, "rt_1_x_jax/b321733791_75882326_000900000")
    seqlen: int = 15
    batch_size: int = 2

    cached: bool = True

    # def __post_init__(self): self.checkpoint_path = osp.join(improve.WEIGHTS, self.checkpoint_path)
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.model, dict):
            self.model = RT1Model(**self.model)
