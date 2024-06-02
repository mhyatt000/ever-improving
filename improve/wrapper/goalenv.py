from __future__ import annotations

import random

import gymnasium as gym
import hydra
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import (ActionWrapper, Env, ObservationWrapper,
                            RewardWrapper, Wrapper)
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.space import Space
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC

import improve
import improve.config.resolver
import improve.wrapper.dict_util as du

class GoalEnvWrapper(ObservationWrapper, RewardWrapper, ActionWrapper, Wrapper):

    def __init__(self, env, goalkey):
        super().__init__(env)
        self.env = env

        # SB3 does not support nested dict
        obspace = env.observation_space
        modspace = {k: v for k, v in obspace.spaces.items() if k != goalkey}
        self.observation_space = Dict(
            {
                **modspace, # all obs besides the goal
                "achieved_goal": obspace[goalkey],
                "desired_goal": obspace[goalkey],
            }
        )

        # TODO
        # this is not a good way to do this
        # ideally have a successful goal
        self.goal = np.zeros_like(obspace[goalkey].sample())
        self.goals = [self.goal]
        self.goal_buffer_size = 10
        self.goalkey = goalkey

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.
        In addition, check if the observation space is correct by inspecting the `observation`, `achieved_goal`, and `desired_goal` keys.
        """
        self.goal = random.choice(self.goals)
        obs, info = self.env.reset(seed=seed)
        obs = self.observation(obs)

        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise Exception(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise Exception(
                    f'GoalEnv requires the "{key}" key to be part of the observation dictionary.'
                )

        return obs, info

    def observation(self, observation):
        observation["achieved_goal"] = observation[self.goalkey]
        del observation[self.goalkey]
        observation["desired_goal"] = self.goal
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.observation(observation)
        reward = self.compute_reward(observation["achieved_goal"], self.goal, info)

        if info["is_success"]:
            self.goals.append(observation["desired_goal"])
            self.goals = self.goals[-self.goal_buffer_size :]

        return observation, reward, terminated, truncated, info


    def compute_reward(self, achieved_goal, desired_goal, info):
        """give small tolerance in case simulator gets weird"""

        if len(achieved_goal.shape) == 4:
            eq = np.isclose(achieved_goal, desired_goal, atol=0.01)
            eq = np.all(eq, axis=(1, 2, 3))
        else:
            eq = np.array(np.allclose(achieved_goal, desired_goal, atol=0.01))
        return eq.astype(np.float32)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return info["is_success"]

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return False  # for now??
