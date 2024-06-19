from __future__ import annotations

import os
import os.path as osp
from datetime import datetime
from pprint import pprint
from typing import Any

import gymnasium as gym
import hydra
import improve
import improve.config.resolver
import improve.wrapper.dict_util as du
import mediapy
import numpy as np
import simpler_env as simpler
import torch
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

import wandb


class EvalWrapper(Wrapper):

    def __init__(self, env, nstep, device, render=True):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.nstep = nstep
        self.device = device
        self.render = render
        self.render_arr = []
        self.n_reset = 0
        self.paths = []

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.render_arr = []
        self.n_reset += 1

        obs, info = self.env.reset(seed=seed, options=options)

        obs["rgb"] = self.image(obs)
        self.maybe_render(obs, False, False)
        return self.to_tensor(obs), info

    def step(self, action):
        action = action.cpu().view(-1).numpy()
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation["rgb"] = self.image(observation)
        self.maybe_render(observation, terminated, truncated, info)
        return self.to_tensor(observation), reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def maybe_render(self, observation, terminated, truncated, info={}):
        if self.render:
            self.render_arr.append(observation["rgb"])
            if terminated or truncated:

                now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                now = datetime.now().strftime("%Y-%m-%d")

                dirname = osp.join(improve.RESULTS, now)
                os.makedirs(osp.dirname(dirname), exist_ok=True)
                path = f"ep_{self.n_reset}_success-{info['success']}.gif"
                path = osp.join(dirname, path)

                mediapy.write_video(path, self.render_arr, fps=5, codec="gif")
                self.paths.append(path)

    def to_wandb(self):
        wandb.log({"eval/video": [wandb.Video(p) for p in self.paths]}, commit=False)
        self.paths = []

    @property
    def instruction(self):
        """returns SIMPLER instruction for ease of use"""
        return self.env.get_language_instruction()

    def image(self, obs):
        """show the right observation for video depending on the robot architecture"""
        image = get_image_from_maniskill2_obs_dict(self.env, obs)
        return image

    def to_tensor(self, data):
        """sends observation to tensor on device"""
        return du.apply(data, lambda x: torch.tensor(x, device=self.device))
