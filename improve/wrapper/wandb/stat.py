import os.path as osp

import gymnasium as gym
from improve.wrapper import dict_util as du


class WandbInfoStatWrapper(gym.Wrapper):
    def __init__(self, env, logger):
        super().__init__(env)
        self.logger = logger

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)

        _info = du.flatten(info, delim="/")
        for k, v in _info.items():
            self.logger.record(osp.join("info", k), v)

        return ob, rew, terminated, truncated, info


class WandbActionStatWrapper(gym.Wrapper):
    """Wrapper that logs action statistics to wandb
    does not support nested dicts in action space
    """

    def __init__(self, env, logger, names, loc="action_stats"):
        super().__init__(env)
        self.logger = logger
        self.names = names
        self.loc = loc

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)

        if len(action) != len(self.names):
            raise ValueError(
                f"Number of action names {len(self.names)} does not match action space {len(action)}"
            )

        for k, v in zip(self.names, action):
            self.logger.record(osp.join("action_stats", k), v)

        return ob, rew, terminated, truncated, info
