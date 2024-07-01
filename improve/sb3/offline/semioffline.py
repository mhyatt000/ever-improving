import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from improve.config.algo import PACCN
# from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

# from stable_baselines3.common.vec_env import VecEnv


class SemiOfflineTrainer:

    def __init__(
        self,
        algocn: PACCN,  # algorithm config node
        modelcn,  # model
        optimcn,  # optimizer config node
        traincn,  # training config node
        loggercn,  # logger config node
        env: Optional[GymEnv] = None,  # environment
    ):

        self.algocn = algocn
        self.modelcn = modelcn
        self.optimcn = optimcn
        self.traincn = traincn
        self.loggercn = loggercn

        self.env = env

        self.setup_model()

    def setup_model(self):

        raise NotImplementedError

        # set up lr schedule
        # set random seed

        # set up rollout buffer
        ## for us this will prob be flex dataloader

        # set up policy (model)
        self.rollout_buffer = self.rollout_buffer_class()
        self.policy = self.policy_class()
        self.policy = self.policy.to(self.device)

    def preprocess_actions(self, actions):
        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                actions = self.policy.unscale_action(actions)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )
        return actions

    def collect_rollouts(self):

        raise NotImplementedError

        self.policy.set_training_mode(False)

        n_steps = 0

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(self.env.num_envs)

        self.callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():

                actions, values, log_probs = self.policy(obs)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            clipped_actions = self.preprocess_actions(clipped_actions)
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.num_timesteps += self.env.num_envs

            # Give access to local variables
            self.callback.update_locals(locals())
            if not self.callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def dump_logs(self):
        pass

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.
        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / elapsed)

        torecord = {
            "time/iterations": iteration,
            "time/fps": fps,
            "time/elapsed": int(elapsed),
            "time/total_timesteps": self.num_timesteps,
            "rollout/ep_rew_mean": safe_mean([ep["r"] for ep in self.ep_info_buffer]),
            "rollout/ep_len_mean": safe_mean([ep["l"] for ep in self.ep_info_buffer]),
            "rollout/success_rate": (
                safe_mean(self.ep_success_buffer)
                if len(self.ep_success_buffer) > 0
                else 0.0
            ),
        }

        for key, value in torecord.items():
            self.logger.record(key, value)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "SemiOfflineAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        """
        do n times
        train the model until it converges or 300k steps
        collect more rollouts
        """

        # self._setup_learn()
        # callback.on_training_start(locals(), globals())
        # self.train()
        # callback.on_training_end()
        # self.collect_rollouts()

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        self.train()

        if self.env is not None:
            while self.num_timesteps < collect_steps:
                continue_training = self.collect_rollouts(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                )

                if not continue_training:
                    break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

        callback.on_training_end()

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
