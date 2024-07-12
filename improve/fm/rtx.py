import copy
from collections import deque
from pprint import pprint
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import simpler_env as simpler
import tensorflow as tf
import tensorflow_hub as hub
from flax.training import checkpoints
from stable_baselines3.common.vec_env import SubprocVecEnv
from transforms3d.euler import euler2axangle

import hydra
import improve
import improve.hydra.resolver
from improve import cn
from improve.fm.cache import load_task_embedding, store_task_embedding
from improve.fm.rt1_model import RT1, detokenize_action
from improve.wrapper import dict_util as du


class RT1Policy:
    """Runs inference with a RT-1 policy."""

    def __init__(
        self,
        model: cn.RT1Model,
        ckpt=None,
        variables=None,
        seqlen=15,
        batch_size=1,
        action_scale=1,
        rng=None,
        policy_setup: str = "google_robot",
        cached=True,
        task=None,
    ):
        """Initializes the policy.

        Args:
          ckpt: A checkpoint point from which to load variables. Either
            this or variables must be provided.
          model: A nn.Module to use for the policy. Must match with the variables
            provided by ckpt or variables.
          variables: If provided, will use variables instead of loading from
            ckpt.
          seqlen: The history length to use for observations.
          rng: a jax.random.PRNGKey to use for the random number generator.
        """


        print(ckpt)
        if not variables and not ckpt:
            raise ValueError("At least one of `variables` or `ckpt` must be defined.")

        self.model = RT1(
            num_image_tokens=model.num_image_tokens,
            num_action_tokens=model.num_action_tokens,
            layer_size=model.layer_size,
            vocab_size=model.vocab_size,
            use_token_learner=model.use_token_learner,
            world_vector_range=model.world_vector_range,
        )

        self._checkpoint_path = ckpt
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.action_scale = action_scale

        self.unnormalize_action = False
        self.unnormalize_action_fxn = None
        self.invert_gripper_action = False
        self.action_rotation_mode = "axis_angle"

        self._run_action_inference_jit = jax.jit(self._run_action_inference)
        # for debugging
        # self._run_action_inference_jit = self._run_action_inference

        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng

        if variables:
            self.variables = variables
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt, None)
            variables = {
                "params": state_dict["params"],
                "batch_stats": state_dict["batch_stats"],
            }
            print("Loaded variables from checkpoint.")
            # print('params', variables['params'])
            # print('batch_stats', variables['batch_stats'])
            self.variables = variables

        # following BatchedOctoInference
        self.hist = deque(maxlen=self.seqlen)
        self.num_image_history = 0

        self.cached = cached
        self.task = task
        if self.cached:
            # tensorflow might be taking the memory?
            # but need it if using T5
            tf.config.experimental.set_visible_devices([], "GPU")

        else:
            self.llm = hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder-large/5"
            )

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()

    def _small_action_filter_google_robot(
        self,
        raw_action: dict[str, np.ndarray | tf.Tensor],
        arm_movement: bool = False,
        gripper: bool = True,
    ) -> dict[str, np.ndarray | tf.Tensor]:
        # small action filtering for google robot
        if arm_movement:
            raw_action["world_vector"] = tf.where(
                tf.abs(raw_action["world_vector"]) < 5e-3,
                tf.zeros_like(raw_action["world_vector"]),
                raw_action["world_vector"],
            )
            raw_action["rotation_delta"] = tf.where(
                tf.abs(raw_action["rotation_delta"]) < 5e-3,
                tf.zeros_like(raw_action["rotation_delta"]),
                raw_action["rotation_delta"],
            )
            raw_action["base_displacement_vector"] = tf.where(
                raw_action["base_displacement_vector"] < 5e-3,
                tf.zeros_like(raw_action["base_displacement_vector"]),
                raw_action["base_displacement_vector"],
            )
            raw_action["base_displacement_vertical_rotation"] = tf.where(
                raw_action["base_displacement_vertical_rotation"] < 1e-2,
                tf.zeros_like(raw_action["base_displacement_vertical_rotation"]),
                raw_action["base_displacement_vertical_rotation"],
            )
        if gripper:
            raw_action["gripper_closedness_action"] = tf.where(
                tf.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                tf.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"],
            )
        return raw_action

    def reset(self, instructions: Optional[List[str]] = None) -> None:

        if not self.cached:
            assert instructions is not None
            self.embeds = []
            for instr in set(instructions): # only the unique ones

                embeds = self.llm([instr]).numpy()
                self.embeds.append(embeds)
                store_task_embedding(embeds, instruction=instr)

                print(embeds.shape)

            print("not cached!")
            self.embeds = np.array(self.embeds)
            print(self.embeds.shape)

        else:
            print("cached!")
            self.embeds = np.array([load_task_embedding(instr) for instr in instructions])
            print(self.embeds.shape)

        self.embeds = np.expand_dims(self.embeds, 1)
        self.embeds = np.repeat(self.embeds, self.seqlen, axis=1)
        self.embeds = np.repeat(self.embeds, self.batch_size, axis=0)

        print(self.embeds.shape)

        self.hist.clear()
        self.num_image_history = 0

    def _run_action_inference(self, observation, rng):
        """A jittable function for running inference."""

        # We add zero action tokens so that the shape is (seqlen, 11).
        # Note that in the vanilla RT-1 setup, where
        # `include_prev_timesteps_actions=False`, the network will not use the
        # input tokens and instead uses zero action tokens, thereby not using the
        # action history. We still pass it in for simplicity.
        act_tokens = jnp.zeros((self.batch_size, 6, 11))

        # Add a batch dim to the observation.
        # batch_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

        _, random_rng = jax.random.split(rng)

        # act = {
        #   "world_vector": jnp.ones((self.batch_size, 15, 3)),
        #   "rotation_delta": jnp.ones((self.batch_size, 15, 3)),
        #   "gripper_closedness_action": jnp.ones((self.batch_size, 15, 1)),
        #   "base_displacement_vertical_rotation": jnp.ones((self.batch_size, 15, 1)),
        #   "base_displacement_vector": jnp.ones((self.batch_size, 15, 2)),
        #   "terminate_episode": jnp.ones((self.batch_size, 15, 3), dtype=jnp.int32),
        # }

        # pprint(du.apply(observation, lambda x: x.shape))

        output_logits = self.model.apply(
            self.variables,
            observation,
            act=None,
            act_tokens=act_tokens,
            train=False,
            rngs={"random": random_rng},
        )

        time_step_tokens = self.model.num_image_tokens + self.model.num_action_tokens
        output_logits = jnp.reshape(
            output_logits, (self.batch_size, self.seqlen, time_step_tokens, -1)
        )
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

        action_logp = jax.nn.softmax(action_logits)
        action_token = jnp.argmax(action_logp, axis=-1)

        # Detokenize the full action sequence.
        detokenized = detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range
        )

        # if self.batch_size == 1:
        # detokenized = jax.tree_map(lambda x: x[0], detokenized)

        return detokenized

    def _add_to_history(self, image: np.ndarray) -> None:
        self.hist.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.seqlen)

    def _obtain_history(self) -> tuple[np.ndarray, np.ndarray]:
        ax = 1  # 0 if self.batch_size == 1 else 1
        observation = np.stack(self.hist, axis=1)  # [: -self.seqlen]
        if observation.shape[1] < self.seqlen:
            pad = np.zeros(
                (
                    self.batch_size,
                    self.seqlen - observation.shape[1],
                    *observation.shape[2:],
                )
            )
            observation = np.concatenate([pad, observation], axis=1)
        return observation[:, -self.seqlen :]

    def step(self, image):
        """Outputs the action given observation from the env."""

        image = copy.deepcopy(image)

        # Resize using TF image resize to avoid any issues with using different
        # resize implementation, since we also use tf.image.resize in the data
        # pipeline. Also scale image to [0, 1].

        # following OXE
        image = tf.image.resize(image, (300, 300)).numpy() / 225.0

        self._add_to_history(image)
        images = self._obtain_history()

        # i think this is for batch? idk
        # obs, pad_mask = obs[None], pad_mask[None]

        observation = {"image": images, "natural_language_embedding": self.embeds}

        self.rng, rng = jax.random.split(self.rng)
        action = self._run_action_inference_jit(observation, rng)
        action = jax.device_get(action)

        """
        # Use the base pose mode if the episode if the network outputs an invalid
        # `terminate_episode` action.
        if np.sum(action["terminate_episode"]) == 0:
            action["terminate_episode"] = np.zeros_like(action["terminate_episode"])
            action["terminate_episode"][-1] = 1
        """

        raw_action = action

        # do the postprocessing itemwise without rewriting everything for now
        a = [du.apply(raw_action, lambda x: x[i]) for i in range(self.batch_size)]
        b = [du.apply(action, lambda x: x[i]) for i in range(self.batch_size)]
        raw_action, action = zip(
            *[self.simpler_postprocessing(_a, _b) for _a, _b in zip(a, b)]
        )
        raw_action, action = du.stack(raw_action), du.stack(action)

        return raw_action, action

    def _unnormalize_action_widowx_bridge(
        self, action: dict[str, np.ndarray | tf.Tensor]
    ) -> dict[str, np.ndarray]:
        action["world_vector"] = self._rescale_action_with_bound(
            action["world_vector"],
            low=-1.75,
            high=1.75,
            post_scaling_max=0.05,
            post_scaling_min=-0.05,
        )
        action["rotation_delta"] = self._rescale_action_with_bound(
            action["rotation_delta"],
            low=-1.4,
            high=1.4,
            post_scaling_max=0.25,
            post_scaling_min=-0.25,
        )
        return action

    @staticmethod
    def _rescale_action_with_bound(
        actions: np.ndarray | tf.Tensor,
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

    def simpler_postprocessing(self, raw_action, action):

        if self.policy_setup == "google_robot":
            raw_action = self._small_action_filter_google_robot(
                raw_action, arm_movement=False, gripper=True
            )
        if self.unnormalize_action:
            raw_action = self.unnormalize_action_fxn(raw_action)
        for k in raw_action.keys():
            raw_action[k] = np.asarray(raw_action[k])

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = (
            np.asarray(raw_action["world_vector"], dtype=np.float64) * self.action_scale
        )
        if self.action_rotation_mode == "axis_angle":
            action_rotation_delta = np.asarray(
                raw_action["rotation_delta"], dtype=np.float64
            )
            action_rotation_angle = np.linalg.norm(action_rotation_delta)
            action_rotation_ax = (
                action_rotation_delta / action_rotation_angle
                if action_rotation_angle > 1e-6
                else np.array([0.0, 1.0, 0.0])
            )

            action["rot_axangle"] = (
                action_rotation_ax * action_rotation_angle * self.action_scale
            )
        elif self.action_rotation_mode in ["rpy", "ypr", "pry"]:
            if self.action_rotation_mode == "rpy":
                roll, pitch, yaw = np.asarray(
                    raw_action["rotation_delta"], dtype=np.float64
                )
            elif self.action_rotation_mode == "ypr":
                yaw, pitch, roll = np.asarray(
                    raw_action["rotation_delta"], dtype=np.float64
                )
            elif self.action_rotation_mode == "pry":
                pitch, roll, yaw = np.asarray(
                    raw_action["rotation_delta"], dtype=np.float64
                )
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action["rot_axangle"] = (
                action_rotation_ax * action_rotation_angle * self.action_scale
            )
        else:
            raise NotImplementedError()

        raw_gripper_closedness = raw_action["gripper_closedness_action"]
        if self.invert_gripper_action:
            # rt1 policy output is uniformized such that -1 is open gripper, 1 is close gripper;
            # thus we need to invert the rt1 output gripper action for some embodiments like WidowX, since for these embodiments -1 is close gripper, 1 is open gripper
            raw_gripper_closedness = -raw_gripper_closedness
        if self.policy_setup == "google_robot":
            # gripper controller: pd_joint_target_delta_pos_interpolate_by_planner; raw_gripper_closedness has range of [-1, 1]
            action["gripper"] = np.asarray(raw_gripper_closedness, dtype=np.float64)
        elif self.policy_setup == "widowx_bridge":
            # gripper controller: pd_joint_pos; raw_gripper_closedness has range of [-1, 1]
            action["gripper"] = np.asarray(raw_gripper_closedness, dtype=np.float64)
            # binarize gripper action to be -1 or 1
            action["gripper"] = 2.0 * (action["gripper"] > 0.0) - 1.0
        else:
            raise NotImplementedError()

        action["terminate_episode"] = raw_action["terminate_episode"]

        # update policy state
        # self.policy_state = policy_step.state

        return raw_action, action
