from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import simpler_env
from gymnasium import logger, spaces
from gymnasium.core import (ActionWrapper, Env, ObservationWrapper,
                            RewardWrapper, Wrapper)
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.space import Space
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict
from pprint import pprint

"""
from gymnasium.envs.registration import (make, make_vec, pprint_registry,
                                         register, register_envs, registry,
                                         spec)
env = gym.make(
    "GraspSingleDummy-v0",
    control_mode=control_mode,
    obs_mode="rgbd",
    robot="widowx_bridge_dataset_camera_setup",
    sim_freq=sim_freq,
    control_freq=control_freq,
    max_episode_steps=50,
    camera_cfgs={"add_segmentation": True},
    rgb_overlay_path=f"ManiSkill2_real2sim/data/real_inpainting/bridge/bridge_{episode_id}_cleanup.png",
    rgb_overlay_cameras=[overlay_camera],
)


def make(task_name):
    assert (
        task_name in ENVIRONMENTS
    ), f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, kwargs = ENVIRONMENT_MAP[task_name]
    kwargs["prepackaged_config"] = True
    env = gym.make(env_name, obs_mode="rgbd", **kwargs)
    return env
"""


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
    """Superclass of wrappers that can modify observations
    using :meth:`observation` for :meth:`reset` and :meth:`step

    uses model (Octo or RTX) to predict initial action
    trained model predicts the partial_action

    example of env.step
    obs, reward, success, truncated, info = env.step(
        np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]]
        ),
    )

    obs_space = {
        "agent": {
            "base_pose": Box(-inf, inf, (7,), float32),
            "controller": {"gripper": {"target_qpos": Box(-inf, inf, (2,), float32)}},
            "qpos": Box(-inf, inf, (11,), float32),
            "qvel": Box(-inf, inf, (11,), float32),
        },
        "camera_param": {
            "base_camera": {
                "cam2world_gl": Box(-inf, inf, (4, 4), float32),
                "extrinsic_cv": Box(-inf, inf, (4, 4), float32),
                "intrinsic_cv": Box(-inf, inf, (3, 3), float32),
            },
            "overhead_camera": {
                "cam2world_gl": Box(-inf, inf, (4, 4), float32),
                "extrinsic_cv": Box(-inf, inf, (4, 4), float32),
                "intrinsic_cv": Box(-inf, inf, (3, 3), float32),
            },
        },
        "extra": {"tcp_pose": Box(-inf, inf, (7,), float32)},
        "image": {
            "base_camera": {
                "Segmentation": Box(0, 4294967295, (128, 128, 4), uint32),
                "depth": Box(0.0, inf, (128, 128, 1), float32),
                "rgb": Box(0, 255, (128, 128, 3), uint8),
            },
            "overhead_camera": {
                "Segmentation": Box(0, 4294967295, (512, 640, 4), uint32),
                "depth": Box(0.0, inf, (512, 640, 1), float32),
                "rgb": Box(0, 255, (512, 640, 3), uint8),
            },
        },
    }
    """

    def __init__(self, env, task, policy, ckpt):
        """Constructor for the observation wrapper."""
        Wrapper.__init__(self, env)

        if policy in ["octo-base", "octo-small"]:
            if ckpt in [None, "None"] or "rt_1_x" in ckpt:
                ckpt = policy

        self.env
        self.task = task
        self.policy = policy
        self.ckpt = ckpt

        model = self.build_model()

        obs, _ = self.env.reset(seed=2022, options=dict(reconfigure=True))
        self.observation_space = convert_observation_to_space(obs)
        self.image_space = convert_observation_to_space(self.get_image(obs))
        self.observation_space.spaces["agent"].spaces["partial_action"] = (
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        )
        self.observation_space = Dict(
            {
                "image": self.image_space,
                # this is just gonna be hardcoded for now
                "partial_action": Box(-np.inf, np.inf, (7,), np.float32),
            }
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

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        # sets seed  to random if it comes from jax
        self.terminated = False
        self.truncated = False
        self.success = False

        obs, info = self.env.reset(seed=seed, options=options)
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

    def pre_step(self):
        pass

    def post_step(self):
        pass

    def step(self, action):
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""

        complete = self.partial + action
        obs, reward, self.success, self.truncated, info = self.env.step(complete)

        obs = self.observation(obs)
        return obs, reward, self.success, self.truncated, info

    def maybe_break(self):
        """returns whether to break the loop"""
        self.truncated = self.terminated or self.truncated
        return self.terminated or self.truncated

    def observation(self, observation):
        """Returns a modified observation."""

        image = self.get_image(observation)

        if self.maybe_break():
            return (
                np.expand_dims(image, axis=0),
                # dont inference the model if env terminates
                np.expand_dims(np.zeros(self.env.action_space.shape), axis=0),
            )

        _, action = self.model.step(image, self.instruction)
        self.terminated = bool(action["terminate_episode"][0] > 0)
        self.maybe_advance()

        observation["agent"]["partial_action"] = action
        self.partial = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]]
        )

        # going to force observation to be just the intended image and partial for now
        obs = (
            np.expand_dims(image, axis=0),
            np.expand_dims(self.partial, axis=0),
        )

        return obs
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


class SB3Wrapper(ResidualRLWrapper):

    def __init__(self, env, task, policy, ckpt):
        super().__init__(env, task, policy, ckpt)

        # define a dict of spaces
        self.observation_space = Dict(
            {
                "image": self.image_space,
                "partial_action": Box(-np.inf, np.inf, (7,), np.float32),
            }
        )

    def observation(self, observation):
        observation = super().observation(observation)
        return {
            "image": observation[0],
            "partial_action": observation[1],
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # for sb3 EvalCallback
        info["is_success"] = info["success"]

        is_bonus = True
        if is_bonus:
            stats = info["episode_stats"].keys()
            bonus = [info[k] for k in stats]
            bonus = [ 0.1 if b else 0 for b in bonus ]
            reward += sum(bonus)

        return observation, reward, terminated, truncated, info


def alldict(thing):
    """gymnasium.spaces.dict.Dict to dict"""
    if type(thing) is gym.spaces.dict.Dict:
        return {k: alldict(v) for k, v in thing.spaces.items()}
    else:
        return thing


def make(task, policy="octo-base", ckpt=None, kind="default"):
    """Creates simulated eval environment from task name."""
    task = simpler_env.ENVIRONMENTS[0] if task is None else task
    env = simpler_env.make(task)
    wrapper = SB3Wrapper if kind == "sb3" else ResidualRLWrapper
    return wrapper(env, task, policy, ckpt)


def main():

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    task = simpler_env.ENVIRONMENTS[0]
    task = "widowx_put_eggplant_in_basket"

    cfg = {
        "task": task,
        "policy": "octo-base",
        "ckpt": None,
    }

    env = make(**cfg, kind="sb3")

    print(env.model)
    print(env.observation_space)
    print()

    hist = {t: 0 for t in simpler_env.ENVIRONMENTS if 'widowx' in t}
    allinstructions = set()
    for t in hist:

        env = make(**cfg, kind="sb3")

        for _ in range(10):
            env.reset()
            for i in range(2000):

                # zero action
                action = np.zeros(env.action_space.shape)
                # random action
                # env.action_space.sample()

                observation, reward, success, truncated, info = env.step(action)

                # print i as formatted for 3 decimals 001 - 100
                print(f"{i:003}", reward, success, truncated)

                allinstructions.add(env.instruction)
                if not (env.instruction in allinstructions):
                    print(env.instruction)

                if truncated : # or success:
                    break

            if success:
                hist[t] += 1
            pprint(hist)

        env.close()
    print(allinstructions)

    

if __name__ == "__main__":
    main()
