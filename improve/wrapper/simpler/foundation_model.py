from __future__ import annotations

from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import (ActionWrapper, Env, ObservationWrapper,
                            RewardWrapper, Wrapper)
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.space import Space
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from scipy.spatial.transform import Rotation as R
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict

import improve.wrapper.dict_util as du


class ExtraObservationWrapper(Wrapper):

    def __init__(self, env, use_image):
        super().__init__(env)

        # other low dim obs
        mk_space = lambda shape: Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
        )

        obs = self.env.observation_space.sample()
        qpos = obs["agent"]["qpos"]

        additions = {
            "agent_qpos-sin": mk_space(qpos.shape),
            "agent_qpos-cos": mk_space(qpos.shape),
            "obj-wrt-eef": mk_space((3,)),
            "eef-pose": mk_space((7,)),
            "obj-pose": mk_space((7,)),
        }

        for k, v in additions.items():
            self.observation_space[k] = v

        self.use_image = use_image
        if use_image:
            image = self.get_image(self.observation_space.sample())
            self.observation_space["simpler-img"] = Box(
                low=0, high=255, shape=image.shape, dtype=np.uint8
            )

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

        if self.use_image:
            observation["simpler-img"] = self.get_image(observation)

        return observation

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, success, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, success, truncated, info

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        # self.obj.pose.transform(self.obj.cmass_local_pose)
        try:
            return self.env.obj.pose
        except:
            return self.env.source_obj_pose

    def get_tcp(self):
        """tool-center point, usually the midpoint between the gripper fingers"""
        eef = self.agent.config.ee_link_name
        tcp = get_entity_by_name(self.agent.robot.get_links(), eef)
        return tcp

    def obj_wrt_eef(self):
        """Get the object pose with respect to the end-effector frame"""
        return self.obj_pose.p - self.get_tcp().pose.p

    def get_image(self, obs):
        """show the right observation for video depending on the robot architecture"""
        image = get_image_from_maniskill2_obs_dict(self.env, obs)
        return image


class ActionSpaceWrapper(Wrapper):
    """modifies the action space.
    rather than masking dimensions, this wrapper changes the visible action space
    it pads the action space with zeros to match the original action space

    helps with model training
    """

    def __init__(self, env, dims):
        super().__init__(env)

        self.dims = dims

        self.old = env.action_space.shape
        low, high = env.action_space.low[0], env.action_space.high[0]
        self.new = (env.action_space.shape[0] - len(dims),)

        self.action_space = Box(low=low, high=high, shape=self.new, dtype=np.float32)

    def action(self, action):
        extra = np.zeros(len(self.dims))
        # must keep the same shape expected by simpler
        action = np.concatenate([action[: self.new[0]], extra])
        return action

    def step(self, action):
        action = self.action(action)
        return self.env.step(action)


class FoundationModelWrapper(Wrapper):
    """
    uses model (Octo or RTX) to predict initial action
    residual policy corrects the initial action

    :param env: environment
    :param task: task name
    :param policy: policy name
    :param ckpt: checkpoint path
    :param residual_scale: residual policy weight
    """

    def __init__(self, env, task, policy, ckpt, residual_scale=1.0, strategy="clip"):
        super().__init__(env)

        if policy in ["octo-base", "octo-small"]:
            if ckpt in [None, "None"] or "rt_1_x" in ckpt:
                ckpt = policy

        self.task = task
        self.policy = policy
        self.ckpt = ckpt
        self.residual_scale = 1.0

        assert strategy in ["dynamic", "clip", None]
        self.strategy = strategy

        translation = np.linalg.norm([0.05, 0.05, 0.05])
        axis, angle = self.rpy_to_axis_angle(*[0.25, 0.25, 0.25])
        self.max = {"translation": translation, "rotation": angle}

        self.bounds = [
            (0.05, -0.05),
            (0.05, -0.05),
            (0.05, -0.05),
            (0.25, -0.25),
            (0.25, -0.25),
            (0.25, -0.25),
            (1, -1),
        ]

        # TODO add option for w_fm and w_rp
        # where w_fm is the weight for the foundation model
        # and w_rp is the weight for the residual policy
        # ie: w_fm + w_rp = 1

        self.build_model()

        self.observation_space["agent_partial-action"] = Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )
        image = self.get_image(self.observation_space.sample())
        self.observation_space["simpler-img"] = Box(
            low=0, high=255, shape=image.shape, dtype=np.uint8
        )

        print(f"{type(self).__name__} initialized")
        print(f"iniitialized {self.policy} model")
        print(f"residual scale: {self.residual_scale}")
        print(f"strategy: {self.strategy}")

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
            from simpler_env.policies.octo.octo_model import OctoInference

            self.model = OctoInference(
                model_type=self.ckpt, policy_setup=policy_setup, init_rng=0
            )

        else:
            raise NotImplementedError()

    def reset(self, **kwargs):
        self.model.reset(self.instruction)
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    @property
    def instruction(self):
        """returns SIMPLER instruction
        for ease of use
        """
        return self.env.get_language_instruction()

    @property
    def final(self):
        """returns whether the current subtask is the final subtask"""
        return self.env.is_final_subtask()

    def get_image(self, obs):
        """show the right observation for video depending on the robot architecture"""
        image = get_image_from_maniskill2_obs_dict(self.env, obs)
        return image

    def compute_final_action(self, action):

        # actions are added together using the rp_scale
        # if the action is out of bounds, it is transformed to be in bounds
        if self.strategy == "clip":
            total_action = self.model_action + (action * self.residual_scale)
            translation = np.linalg.norm(total_action[:3])
            axis, rotation = self.rpy_to_axis_angle(*total_action[3:6])

            # dont go out of bounds
            if abs(translation) > self.max["translation"]:
                print("OOB translation", total_action[:3])
                total_action[:3] = total_action[:3] * (
                    self.max["translation"] / translation
                )
                print(total_action[:3])
            if abs(rotation) > self.max["rotation"]:
                print("OOB rotation", total_action[3:6])
                total_action[3:6] = self.axis_angle_to_rpy(axis, self.max["rotation"])
                print(total_action[3:6])

        # residual actions transformed to the remaining action space after FM
        # added together without rp_scale
        if self.strategy == "dynamic":
            bounds = [
                (high - a, low - a)
                for (high, low), a in zip(self.bounds, self.model_action)
            ]

            def f(x, b):
                return asymmetric_transform(
                    x,
                    low=-1,
                    high=1,
                    post_scaling_max=b[0],
                    post_scaling_min=b[1],
                )

            action = np.array([f(a, b) for a, b in zip(action, bounds)])
            return action + self.model_action

        if self.strategy is None:
            total_action = self.model_action + (action * self.residual_scale)
            return total_action

    def step(self, action):

        total_action = self.compute_final_action(action)
        obs, reward, success, truncated, info = self.env.step(total_action)
        # dont compute this
        # print('WARNING: using RPL action penalty')
        # reward = self.compute_reward(action, reward)

        obs = self.observation(obs)
        self.image = self.get_image(obs)
        info["agent_partial-action"] = {
            k: v
            for k, v in zip(
                ["x", "y", "z", "y", "p", "r", "g"], self.unscale(self.model_action)
            )
        }
        return obs, reward, success, truncated, info

    def compute_reward(self, action, reward):
        reward - (1e-4 * np.linalg.norm(self.unscale(action)))
        return reward

    def observation(self, observation):
        """Returns a modified observation."""

        image = self.get_image(observation)

        _, action = self.model.step(image, self.instruction)
        # self.maybe_advance()

        action = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]]
        )
        self.model_action = action
        observation["agent_partial-action"] = self.unscale(action)

        return observation

    @staticmethod
    def rpy_to_axis_angle(roll, pitch, yaw):

        rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        axis_angle = rotation.as_rotvec()

        # The angle is the magnitude of the rotation vector
        angle = np.linalg.norm(axis_angle)

        # The axis is the normalized rotation vector
        # This should be [0, 0, 0] if there is no rotation
        axis = axis_angle / angle if angle != 0 else axis_angle

        return axis, angle

    def axis_angle_to_rpy(self, axis, angle):
        rotation = R.from_rotvec(axis * angle)
        rpy = rotation.as_euler("xyz", degrees=False)
        return rpy

    def maybe_advance(self):
        """advance the environment to the next subtask"""
        if self.terminated and (not self.final):
            self.terminated = False
            self.env.advance_to_next_subtask()

    @property
    def final(self):
        """returns whether the current subtask is the final subtask"""
        return self.env.is_final_subtask()

    def to(self, device):
        raise NotImplementedError()

        if device and self.model is not None:
            params = jax.device_put(self.model.params, jax.devices("gpu")[device])
            del self.model.params
            self.model.params = params
        self.device = device

    def close(self):

        # deallocate model
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

    def unscale(self, action):
        """unscales the action"""
        return preprocess_action(action)


def preprocess_action(action: np.ndarray) -> np.ndarray:
    action = {
        "world_vector": action[:3],
        "rotation_delta": action[3:6],
        "gripper": action[-1],
    }

    action = _unnormalize_rtx_for_observation(action)
    action = np.concatenate(
        [
            action["world_vector"],
            action["rotation_delta"],
            np.array([action["gripper"]]),
        ]
    )
    return action


def _rescale_action_with_bound(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0.0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return np.clip(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def _unnormalize_rtx_for_observation(
    action: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    action["world_vector"] = _rescale_action_with_bound(
        action["world_vector"],
        low=-0.05,
        high=0.05,
        post_scaling_max=1.75,
        post_scaling_min=-1.75,
    )
    action["rotation_delta"] = _rescale_action_with_bound(
        action["rotation_delta"],
        low=-0.25,
        high=0.25,
        post_scaling_max=1.4,
        post_scaling_min=-1.4,
    )
    return action


def asymmetric_transform(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0.0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    """Modified rescale function ensuring 0 maps to 0."""

    # dont divide by zero
    pos_scale = post_scaling_max / max(high, 1e-8)
    neg_scale = post_scaling_min / min(low, -1e-8)

    resc_actions = np.where(actions >= 0, actions * pos_scale, actions * neg_scale)
    return np.clip(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )
