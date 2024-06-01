from __future__ import annotations

import random
from pprint import pprint
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import simpler_env
from gymnasium import logger, spaces
from gymnasium.core import (ActionWrapper, Env, ObservationWrapper,
                            RewardWrapper, Wrapper)
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.space import Space
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from scipy.ndimage import zoom
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict

import improve
import improve.config.resolver


class GoalEnv(ObservationWrapper, RewardWrapper, ActionWrapper, Wrapper):

    def __init__(self, env, goalkey):
        super().__init__(env)
        self.env = env

        obspace = env.observation_space
        self.observation_space = Dict(
            {
                "observation": obspace,
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.
        In addition, check if the observation space is correct by inspecting the `observation`, `achieved_goal`, and `desired_goal` keys.
        """
        self.goal = random.choice(self.goals)
        things = self.env.reset(seed=seed)

        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    f'GoalEnv requires the "{key}" key to be part of the observation dictionary.'
                )
        return things

    def observation(self, observation):
        observation = self.env.observation(observation)
        return {
            "observation": observation,
            "achieved_goal": observation[self.goalkey],
            "desired_goal": self.goal,
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.observation(observation)
        reward = self.compute_reward(observation["achieved_goal"], self.goal, info)

        if info["is_success"]:
            self.goals.append(observation["desired_goal"])
            self.goals = self.goals[-self.goal_buffer_size :]

    def compute_reward(self, achieved_goal, desired_goal, info):
        """give small tolerance in case simulator gets weird"""
        eq = np.allclose(achieved_goal, desired_goal, atol=0.01)
        return 1 if eq else 0

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return info["is_success"]

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return False  # for now??
