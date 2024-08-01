from collections import deque
from functools import partial
from typing import List, Optional

import improve
import jax
import numpy as np
import tensorflow as tf
from improve.wrapper import dict_util as du
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from octo.model.octo_model import OctoModel
from octo.utils.train_utils import freeze_weights, merge_params
from simpler_env.policies.octo.octo_model import OctoInference
from simpler_env.utils.action.action_ensemble import ActionEnsembler
from transforms3d.euler import euler2axangle

"""
# create a 1D mesh with a single axis named "batch"
mesh = Mesh(jax.devices(), axis_names="batch")
# Our batches will be data-parallel sharded -- each device will get a slice of the batch
dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
# Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
replicated_sharding = NamedSharding(mesh, PartitionSpec())
"""


class BatchedOctoInference(OctoInference):
    def __init__(
        self,
        batch_size: int = 8,
        cached: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        print(type(self))
        print(f"batch_size: {batch_size}")

        if batch_size > 1:
            example_batch = du.apply(
                self.model.example_batch, lambda x: np.concatenate([x] * batch_size)
            )
        else:
            example_batch = self.model.example_batch

        new = OctoModel.from_config(
            self.model.config,
            example_batch,
            self.model.text_processor,
            verbose=True,
            # dataset_statistics=dataset.dataset_statistics,
        )

        merged_params = merge_params(new.params, self.model.params)
        new = new.replace(params=merged_params)
        del self.model
        self.model = new

        # TODO
        # self.policy_setup = policy_setup
        # from SIMPLER OctoInference policy_setup
        self.action_ensemble = True
        self.action_ensemble_temp = 0.0
        if self.action_ensemble:
            self.action_ensembler = BatchedActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp, self.batch_size
            )
        else:
            self.action_ensembler = None

        self.cached = cached
        if self.cached:
            del self.model.tokenizer
        tf.config.experimental.set_visible_devices([], "GPU")

        """
        self.fwd = jax.jit(
            fun=self._fwd,
            in_shardings=[replicated_sharding, dp_sharding],
            out_shardings=(replicated_sharding, replicated_sharding),
            donate_argnums=0,
        )
        """
        self.fwd = self._fwd

    def reset(self, descs: List[str]) -> None:
        self.reset_all(descs)
        # use cached embeds
        # del self.model.tokenizer

    def reset_all(self, descs: List[str]) -> None:
        if self.automatic_task_creation:
            self.task = self.model.create_tasks(texts=descs)
        else:
            self.task = self.tokenizer(descs, **self.tokenizer_kwargs)

        self.descs = descs
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = np.full((self.batch_size,), False)
        self.gripper_action_repeat = np.full((self.batch_size,), 0)
        self.sticky_gripper_action = np.full((self.batch_size,), 0.0)
        # parent removed this ...
        # self.gripper_is_closed = False
        self.previous_gripper_action = np.full((8,), np.nan)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        ax = 1  # 0 if self.batch_size == 1 else 1
        images = np.stack(self.image_history, axis=ax)
        horizon = len(self.image_history)
        # note: this should be of float type, not a bool type
        pad_mask = np.ones(horizon, dtype=np.float64)
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this should be of float type, not a bool type
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask

    def _fwd(self, model, images, pad_mask, task, automatic_task_creation, rng, key):
        """only for DDP speedup"""

        if automatic_task_creation:
            input_observation = {"image_primary": images, "pad_mask": pad_mask}
            norm_raw_actions = model.sample_actions(input_observation, task, rng=key)

        else:
            input_observation = {"image_primary": images, "timestep_pad_mask": pad_mask}
            input_observation = {
                "observations": input_observation,
                "tasks": {"language_instruction": task},
                "rng": np.concatenate([rng, key]),
            }
            norm_raw_actions = model.lc_ws2(input_observation)[:, :, :7]

        return norm_raw_actions

    def step(
        self, image: np.ndarray, descs: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (B, H, W, 3), uint8
            descs: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if descs is not None:
            if descs != self.descs:
                # task description has changed; reset the policy state
                self.reset(descs)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()

        pad_mask = np.repeat(pad_mask[None, :], self.batch_size, axis=0)

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]
        # print("octo local rng", self.rng, key)

        norm_raw_actions = self.fwd(
            self.model,
            images,
            pad_mask,
            self.task,
            self.automatic_task_creation,
            self.rng,
            key,
        )

        assert norm_raw_actions.shape == (self.batch_size, self.pred_action_horizon, 7)

        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)

        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        raw_action = {
            "world_vector": np.array(raw_actions[:, :3]),
            "rotation_delta": np.array(raw_actions[:, 3:6]),
            "open_gripper": np.array(
                raw_actions[:, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )

        axangles = []
        for rotation_delta in action_rotation_delta:
            roll, pitch, yaw = rotation_delta
            ax, angle = euler2axangle(roll, pitch, yaw)
            axangle = ax * angle
            axangles.append(axangle[None])

        action["rot_axangle"] = np.concatenate(axangles, axis=0) * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # This is one of the ways to implement gripper actions; we use an alternative implementation below for consistency with real
            # gripper_close_commanded = (current_gripper_action < 0.5)
            # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

            # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
            # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
            #     self.sticky_action_is_on = True
            #     self.sticky_gripper_action = relative_gripper_action

            # if self.sticky_action_is_on:
            #     self.gripper_action_repeat += 1
            #     relative_gripper_action = self.sticky_gripper_action

            # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
            #     self.sticky_action_is_on = False
            #     self.gripper_action_repeat = 0

            # action['gripper'] = np.array([relative_gripper_action])

            # wow. what a mess when vectorized
            # alternative implementation
            relative_gripper_action = np.where(
                np.isnan(self.previous_gripper_action),
                np.zeros_like(self.previous_gripper_action),
                self.previous_gripper_action - current_gripper_action,
            )
            self.previous_gripper_action = current_gripper_action

            to_stick = np.logical_and(
                np.abs(relative_gripper_action) > 0.5,
                (self.sticky_action_is_on is False),
            )
            self.sticky_action_is_on = np.where(
                to_stick, True, self.sticky_action_is_on
            )
            self.sticky_gripper_action = np.where(
                to_stick, relative_gripper_action, self.sticky_gripper_action
            )

            self.gripper_action_repeat += self.sticky_action_is_on.astype(int)
            relative_gripper_action = np.where(
                self.sticky_action_is_on,
                self.sticky_gripper_action,
                relative_gripper_action,
            )

            reset = self.gripper_action_repeat == self.sticky_gripper_num_repeat
            self.sticky_action_is_on = np.where(reset, False, self.sticky_action_is_on)
            self.gripper_action_repeat = np.where(reset, 0, self.gripper_action_repeat)
            self.sticky_gripper_action = np.where(
                reset, 0.0, self.sticky_gripper_action
            )

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            # binarize gripper action to 1 (open) and -1 (close)
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0] * self.batch_size)

        return raw_action, action


class BatchedActionEnsembler(ActionEnsembler):
    def __init__(self, pred_action_horizon, action_ensemble_temp=0.0, batch_size=1):
        self.ensemblers = [
            ActionEnsembler(pred_action_horizon, action_ensemble_temp)
            for _ in range(batch_size)
        ]

    def reset(self):
        for ensembler in self.ensemblers:
            ensembler.reset()

    def ensemble_action(self, cur_action):
        return np.stack(
            [
                ensembler.ensemble_action(cur_action[i])
                for i, ensembler in enumerate(self.ensemblers)
            ]
        )
