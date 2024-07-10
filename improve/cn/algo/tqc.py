from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from hydra.core.config_store import ConfigStore
from improve.cn.algo.base import Algo
from improve.cn.algo.sac import SAC, OffPolicy
from improve.util.config import default, store


@dataclass
class TQC(OffPolicy):
    name = "tqc"

    gradient_steps = 4  # default is 1 # UTD in DEAC paper

    # defined in SAC?
    # target_entropy = -7  # 'auto'

    top_quantiles_to_drop_per_net = 0  # 2 is default

    policy_kwargs = default(
        dict(
            share_features_extractor=True,
            n_quantiles=35,  # 25 is default
            n_critics=10,  # from DEAC ... 2 is default
        )
    )
