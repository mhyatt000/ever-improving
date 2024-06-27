import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Union

from hydra.core.config_store import ConfigStore

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


@store
@dataclass
class PPOCN:
    name: str = "ppo"

    policy_kwargs: dict = default({"share_features_extractor": True})

    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    target_kl: Optional[float] = None

    learning_rate: float = 3e-4
    batch_size: int = 256
    n_steps: int = 2048

    stats_window_size: int = 10  # was 100 ... for logging only
    seed: int = 0


# use_original_space: False
# warmup_zero_action: False

# Using the type
# cs.store(name="ppo", node=PPOCN)
