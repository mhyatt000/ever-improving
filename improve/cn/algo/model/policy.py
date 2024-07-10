from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from improve.cn.algo.base import Algo
from improve.cn.algo.sac import SAC, OffPolicy
from improve.util.config import default, store


class Activation(Enum):
    ReLU = "relu"
    Tanh = "tanh"
    LeakyReLU = "leakyrelu"
    PReLU = "prelu"
    ELU = "elu"
    SELU = "selu"
    Swish = "swish"
    Mish = "mish"
    GELU = "gelu"
    Softmax = "softmax"
    NONE = None


class Extractor(Enum):
    # Mlp = "MLP"
    # Cnn = "CNN"
    NatureCNN = "nature_cnn"
    # ImpalaCNN = "impala_cnn"
    # CustomCNN = "custom_cnn"
    # CustomMLP = "custom_mlp"
    CombinedExtractor = "CombinedExtractor"


@dataclass
class Policy:
    lr_schedule: Optional[Any] = None
    net_arch: Optional[Any] = None
    activation_fn: Optional[Activation] = Activation.Tanh
    ortho_init: bool = False
    use_sde: bool = False
    log_std_init: float = 0.0
    full_std: bool = True
    use_expln: bool = False
    squash_output: bool = False
    features_extractor_class: Extractor = Extractor.CombinedExtractor
    features_extractor_kwargs: Optional[Any] = None
    share_features_extractor: bool = True
    normalize_images: bool = True
