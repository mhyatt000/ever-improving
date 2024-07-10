from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from improve.cn.algo.base import Algo
from improve.util.config import default, store

@store
@dataclass
class Buffer:
    name: str = "base"

    cls: Optional[str] = None
    args: Optional[Any] = None
    size: int = int(1e6)


"""
# HER hindsight experience replay buffer
# works with DQN, SAC, DDPG and TD3

name: "her"
cls: ${r_typeof:stable_baselines3.HerReplayBuffer}


# _target_: "stable_baselines3.HerReplayBuffer"

size: ${r_toint:1e3}
args:
  # Available strategies (cf paper): future, final, episode
  goal_selection_strategy: "future" # equivalent to GoalSelectionStrategy.FUTURE
  n_sampled_goal: 4
# device: 'cpu'

# observation_space: null
# action_space: null
# env: null
# n_envs: 1
"""
