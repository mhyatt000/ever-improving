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
import wandb
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


def alldict(thing):
    """gymnasium.spaces.dict.Dict to dict"""
    if type(thing) is gym.spaces.dict.Dict:
        return {k: alldict(v) for k, v in thing.spaces.items()}
    else:
        return thing


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

    def __init__(
        self,
        env,
        task,
        policy,
        ckpt,
        use_original_space=False,
        residual_scale=1,
        force_seed=False,
        seed=None,
    ):
        """Constructor for the observation wrapper."""
        Wrapper.__init__(self, env)

        if policy in ["octo-base", "octo-small"]:
            if ckpt in [None, "None"] or "rt_1_x" in ckpt:
                ckpt = policy

        self.env
        self.task = task
        self.policy = policy
        self.ckpt = ckpt

        self.use_original_space = use_original_space
        self.residual_scale = residual_scale

        self.force_seed = force_seed
        self.seed = seed

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

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        # sets seed  to random if it comes from jax
        self.terminated = False
        self.truncated = False
        self.success = False

        if self.force_seed:
            print(f"forcing seed to {self.seed}")
            seed = self.seed

        obs, info = self.env.reset(seed=seed, options=options)
        if self.model is not None:
            self.model.reset(self.instruction)

        return self.observation(obs), info

    @property
    def final(self):
        """returns whether the current subtask is the final subtask"""
        return self.env.is_final_subtask()

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        # self.obj.pose.transform(self.obj.cmass_local_pose)
        return self.env.obj_pose

    def get_tcp(self):
        """tool-center point, usually the midpoint between the gripper fingers"""
        eef = self.agent.config.ee_link_name
        tcp = get_entity_by_name(self.agent.robot.get_links(), eef)
        return tcp

    def obj_wrt_eef(self):
        """Get the object pose with respect to the end-effector frame"""
        return self.obj_pose.p - self.get_tcp().pose.p

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

        # if learning in the original space, the partial is added inside algo
        if not self.use_original_space:
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

        # add sin and cos of qpos
        qpos = observation["agent"]["qpos"]
        observation["agent"]["qpos-sin"] = np.sin(qpos)
        observation["agent"]["qpos-cos"] = np.cos(qpos)
        # eef and obj pose
        tcp, obj = self.get_tcp().pose, self.obj_pose
        observation["eef-pose"] = np.array([*tcp.p, *tcp.q])
        observation["obj-pose"] = np.array([*obj.p, *obj.q])
        observation["obj-wrt-eef"] = np.array(self.obj_wrt_eef())

        image = self.get_image(observation)
        observation["simpler-img"] = image

        # early return if no model
        if self.model is None:
            self.partial = np.zeros(7)
            observation["agent"]["partial-action"] = self.partial
            return observation

        if self.maybe_break():
            pass  # this is fine for now

        _, action = self.model.step(image, self.instruction)
        self.terminated = bool(action["terminate_episode"][0] > 0)
        self.maybe_advance()

        self.partial = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]]
        )
        observation["agent"]["partial-action"] = self.partial

        """
        # going to force observation to be just the intended image and partial for now
        # update: add qpos and qvel for PPO
        obs = (
            np.expand_dims(image, axis=0),
            np.expand_dims(self.partial, axis=0),
        )
        # return obs
        """

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

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 5}

    def __init__(
        self,
        env,
        task,
        policy,
        ckpt,
        use_wandb=False,
        downscale=None,
        device=None,
        keys=None,
        use_original_space=False,
        residual_scale=1,
        force_seed=False,
        seed=None,
        _reward_type="sparse",
    ):
        super().__init__(
            env,
            task,
            policy,
            ckpt,
            use_original_space,
            residual_scale,
            force_seed,
            seed,
        )

        self.use_wandb = use_wandb
        self._reward_type = _reward_type
        self.downscale = downscale

        if device and self.model is not None:
            params = jax.device_put(self.model.params, jax.devices("gpu")[device])
            del self.model.params
            self.model.params = params
        self.device = device

        # filter for the desired obs space
        spaces = du.flatten(alldict(self.observation_space))
        spaces = {k: v for k, v in spaces.items() if k in keys}
        self.keys = keys

        self.observation_space = Dict(spaces)
        if self.downscale and "simpler-img" in self.keys:

            sample = self.observation_space.spaces["simpler-img"].sample()
            shape = self.scale_image(
                sample,
                1 / self.downscale,
            ).shape
            dtype = sample.dtype
            self.observation_space.spaces["simpler-img"] = gym.spaces.Box(
                low=0, high=255, shape=shape, dtype=dtype
            )

        self.image = None
        self.images = []
        self.render_every = 55
        self.render_counter = 0

    def finish_render(self):
        if self.images and self.use_wandb:
            n = self.render_counter % self.render_every
            now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            now = datetime.now().strftime("%Y-%m-%d")

            dirname = osp.join(improve.RESULTS, now)
            # os.makedirs(osp.dirname(dirname), exist_ok=True)
            path = f"ep_{n}_success-.mp4"
            # path = osp.join(dirname, path)

            mediapy.write_video(path, self.images, fps=5)

            wandb.log(
                {f"video/buffer{n}": wandb.Video(path, fps=5)},
                # step=self.nstep,
            )
            self.images = []

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.images = []
        self.render_counter += 1
        return super().reset(seed=seed, options=options)

    def scale_image(self, image, scale):

        # TODO can we get rid of batch dim?
        zoom_factors = (scale, scale, 1)
        scaled_image = zoom(image, zoom_factors)
        return scaled_image

    def observation(self, observation):
        observation = super().observation(observation)
        observation = du.flatten(alldict(observation))
        self.image = observation["simpler-img"]  # for render only
        observation = {k: v for k, v in observation.items() if k in self.keys}

        if self.downscale and "simpler-img" in self.keys:
            observation["simpler-img"] = self.scale_image(
                self.image, 1 / self.downscale
            )

        return observation

    def render(self, mode=None):
        if mode == "human":
            plt.imshow(self.image)
            plt.title("SIMPLER")
            plt.pause(0.1)
        else:
            self.images.append(self.image)
            return self.image

    def compute_reward(self, observation, action, reward, terminated, truncated, info):
        """Compute the reward."""

        if self._reward_type == "sparse":
            return reward

        if self._reward_type == "transic":  # but customized
            # small penalty for large actions/join velocities (1e-5)
            # 0.1 * reward for shorter distance/lifting object
            # large reward for succeeding
            return (
                200 * reward
                + ((0.1) * (1 - torch.tanh(10 * np.linalg.norm(observation["obj-wrt-eef"]))))
                + int(info["lifted_object"])
                + (0.25 * info["is_grasped"])
                + ((1e-3) * sum(observation["q_vel"]))
                + ((1e-3) * sum(action) if self.model is not None else 0) # RP shouldnt help too much
            )

        if self._reward_type == "robosuite":
            # reward function from robosuite cube pickup task
            if info["success"]:
                return 2.25
            else:
                return (
                    1
                    - np.tanh(10.0 * np.linalg.norm(observation["obj-wrt-eef"]))
                    + (0.25 * info["is_grasped"])
                    + (info["lifted_object"])
                )  # additional lifting reward

        return reward

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward = self.compute_reward(
            observation, action, reward, terminated, truncated, info
        )

        # for sb3 EvalCallback
        info["is_success"] = info["success"]
        return observation, reward, terminated, truncated, info

        # for sb3 EvalCallback
        info["is_success"] = info["success"]

        self.render()
        if terminated or truncated and (self.render_counter % self.render_every) < 10:
            self.finish_render()

        return observation, reward, terminated, truncated, info

    def close(self):

        # deallocate model
        if self.model is not None:
            del self.model
            self.model = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # manually call garbage collector for jax models
            # might not be necessary
            import gc

            gc.collect()

            import tensorflow as tf

            tf.keras.backend.clear_session()

        super().close()


def make(cn):
    """Creates simulated eval environment from task name.
    param: cn: config node
    """
    env = simpler_env.make(
        cn.task,
        # **{ "renderer_kwargs": { "device": "cuda:0", "offscreen_only": True, } },
    )
    wrapper = SB3Wrapper if cn.kind == "sb3" else ResidualRLWrapper

    return wrapper(
        env,
        cn.task,
        cn.foundation.name,
        cn.foundation.ckpt,
        use_wandb=cn.use_wandb,
        downscale=cn.downscale,
        keys=cn.obs_keys,
        use_original_space=cn.use_original_space,
        force_seed=cn.seed.force,
        seed=cn.seed.value if cn.seed.force else None,
        _reward_type=cn.reward,
    )


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    pprint(OmegaConf.to_container(cfg, resolve=True))

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    env = make(cfg.env)
    obs, info = env.reset()

    eefs, objs, dists = [], [], []
    for _ in tqdm(range(10)):

        randoms = env.action_space.sample()
        zeros = np.zeros(env.action_space.shape)
        zeros[-1] = 0.1

        observation, reward, success, truncated, info = env.step(zeros)
        print(observation)

    quit()
    # print(info)

    for i in tqdm(range(120)):

        zeros = np.zeros(env.action_space.shape)
        randoms = env.action_space.sample()
        observation, reward, success, truncated, info = env.step(randoms)

        # print(f"{i:003}", reward, success, truncated)
        eefs.append(env.get_tcp().pose.p)
        objs.append(env.obj_pose.p)
        dists.append(env.obj_wrt_eef())

        if truncated:  # or success:
            break

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    eefs = np.array(eefs)
    objs = np.array(objs)
    dists = np.array(dists)
    names = ["x", "y", "z"]

    for i in range(3):
        axs[i].plot(eefs[:, i], label="eef")
        axs[i].plot(objs[:, i], label="obj")
        axs[i].plot(dists[:, i], label="obj wrt eef")

        axs[i].set_title(names[i])
        axs[i].legend()

        axs[i].axhline(0, color="red", linestyle="--")
        # axs[i].set_ylim(-1, 1)

    env.close()
    plt.savefig("obj_wrt_eef.png")
    # plt.show()


if __name__ == "__main__":
    main()
