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
import simpler_env
from gymnasium import logger, spaces
from gymnasium.core import (ActionWrapper, Env, ObservationWrapper,
                            RewardWrapper, Wrapper)
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.space import Space
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from scipy.ndimage import zoom
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict
from tqdm import tqdm

import wandb


# ---------------------------------------------------------------------------- #
# OpenAI gym
# Maniskill2
# ---------------------------------------------------------------------------- #
def get_dtype_bounds(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.bool_):
        return 0, 1
    else:
        raise TypeError(dtype)


def convert_observation_to_space(observation, prefix=""):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        # CATUION: Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        space = spaces.Dict(
            [
                (k, convert_observation_to_space(v, prefix + "/" + k))
                for k, v in observation.items()
            ]
        )
    elif isinstance(observation, np.ndarray):
        shape = observation.shape
        dtype = observation.dtype
        low, high = get_dtype_bounds(dtype)
        if np.issubdtype(dtype, np.floating):
            low, high = -np.inf, np.inf
        space = spaces.Box(low, high, shape=shape, dtype=dtype)
    elif isinstance(observation, (float, np.float32, np.float64)):
        logger.debug(f"The observation ({prefix}) is a (float) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, (int, np.int32, np.int64)):
        logger.debug(f"The observation ({prefix}) is a (integer) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=int)
    elif isinstance(observation, (bool, np.bool_)):
        logger.debug(f"The observation ({prefix}) is a (bool) scalar")
        space = spaces.Box(0, 1, shape=[1], dtype=np.bool_)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class ResidualRLWrapper(ObservationWrapper):

    def __init__( self, env, task, policy, ckpt, residual_scale=1):
        super().__init__(self, env)

        if policy in ["octo-base", "octo-small"]:
            if ckpt in [None, "None"] or "rt_1_x" in ckpt:
                ckpt = policy

        self.env
        self.task = task
        self.policy = policy
        self.ckpt = ckpt

        self.residual_scale = residual_scale

        model = self.build_model()

        obs, _ = self.env.reset(options=dict(reconfigure=True))
        self.observation_space = convert_observation_to_space(obs)

        self.image_space = convert_observation_to_space(self.get_image(obs))
        self.observation_space.spaces["simpler-img"] = self.image_space

        # other low dim obs
        qpos = obs["agent"]["qpos"]
        self.observation_space.spaces["agent"]["qpos-sin"] = Box(
            low=-np.inf, high=np.inf, shape=qpos.shape, dtype=np.float32
        )
        self.observation_space.spaces["agent"]["qpos-cos"] = Box(
            low=-np.inf, high=np.inf, shape=qpos.shape, dtype=np.float32
        )

        self.observation_space.spaces["obj-wrt-eef"] = Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space.spaces["eef-pose"] = Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space.spaces["obj-pose"] = Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # agent partial action
        self.observation_space.spaces["agent"].spaces["partial-action"] = (
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        )

    def build_model(self):
        """Builds the model."""

        # build policy
        if "google_robot" in self.task:
            policy_setup = "google_robot"
        elif "widowx" in self.task:
            policy_setup = "widowx_bridge"
        else:
            raise NotImplementedError()

        self.model = None
        if self.policy is not None:
            if self.policy == "rt1":
                from simpler_env.policies.rt1.rt1_model import RT1Inference

                self.model = RT1Inference(
                    saved_model_path=self.ckpt, policy_setup=policy_setup
                )

            elif "octo" in self.policy:
                from improve.simpler_mod.octo import OctoInference

                self.model = OctoInference(
                    model_type=self.ckpt, policy_setup=policy_setup, init_rng=0
                )

            else:
                raise NotImplementedError()

    def reset(self, **kwargs):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        self.terminated = False
        self.truncated = False
        self.success = False

        obs, info = self.env.reset(**kwargs)
        if self.model is not None:
            self.model.reset(self.instruction)

        return self.observation(obs), info

    @property
    def final(self):
        """returns whether the current subtask is the final subtask"""
        return self.env.is_final_subtask()

    @property
    def instruction(self):
        """returns SIMPLER instruction
        for ease of use
        """
        return self.env.get_language_instruction()

    def step(self, action):
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""

        action = self.partial + action * self.residual_scale

        obs, reward, self.success, self.truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, self.success, self.truncated, info

    def maybe_break(self):
        """returns whether to break the loop"""
        self.truncated = self.terminated or self.truncated
        return self.terminated or self.truncated

    def observation(self, observation):
        """Returns a modified observation."""

        image = self.get_image(observation)
        observation["simpler-img"] = image

        if self.maybe_break():
            pass  # this is fine for now

        _, action = self.model.step(image, self.instruction)
        self.terminated = bool(action["terminate_episode"][0] > 0)
        self.maybe_advance()

        self.partial = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]]
        )
        observation["agent"]["partial-action"] = self.partial

        return observation  # just the tuple gets returned for now

    def get_image(self, obs):
        """show the right observation for video depending on the robot architecture"""
        image = get_image_from_maniskill2_obs_dict(self.env, obs)
        return image

    def maybe_advance(self):
        """advance the environment to the next subtask"""
        if self.terminated and (not self.final):
            self.terminated = False
            self.env.advance_to_next_subtask()


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    pprint(OmegaConf.to_container(cfg, resolve=True))

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    env = make(cfg.env)
    obs, info = env.reset()


if __name__ == "__main__":
    main()
