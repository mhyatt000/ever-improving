import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Union

from hydra.core.config_store import ConfigStore
from improve import CS

# from improve import CS


cs = ConfigStore.instance()


def store(cls):
    """
    @store will call
    cs.store(node=<class type>, name=<filename with no extension>, group=<dirname>)
    """

    def wrapper(cls):
        tree = inspect.getfile(cls).split(".")[0].split("/")
        name = tree[-1]
        group = tree[-2]
        print(name, group)
        cs.store(name=name, node=cls, group=group)
        return cls

    return wrapper(cls)


def default(data):
    return field(default_factory=lambda: data)

class BaseAlgoCN:
    stats_window_size: int = 10  # was 100
    tensorboard_log: str = None
    seed: int = 0
    device: str = "auto"
    _init_setup_model: bool = True

@store
@dataclass
class PACCN:
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


@store
@dataclass
class TrainerCN:
    learning_rate: float = 3e-4
    batch_size: int = 256
    device: str = "auto"
    seed: int = 0


@store(name="base", group="log")
@dataclass
class LoggerCN:
    stats_window_size: int = 10  # was 100
    tensorboard_log: str = None
    _init_setup_model: bool = True
