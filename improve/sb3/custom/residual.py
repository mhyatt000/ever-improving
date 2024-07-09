import io
import copy
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   RolloutReturn, Schedule,
                                                   TrainFreq,
                                                   TrainFrequencyUnit)
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from improve.config import FoundationModel_CN, OctoS_CN
from improve.env import ActionRescaler
from improve.fm import build_foundation_model
from improve.sb3.custom import CHEF


@dataclass
class Algo_CN:
    learning_rate: Union[float, Schedule]
    buffer_size: int = 1_000_000  # 1e6
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: Union[int, Tuple[int, str]] = (1, "step")
    gradient_steps: int = 1
    action_noise: Optional[ActionNoise] = None
    replay_buffer_class: Optional[Type[ReplayBuffer]] = None
    replay_buffer_kwargs: Optional[Dict[str, Any]] = None
    optimize_memory_usage: bool = False
    policy_kwargs: Optional[Dict[str, Any]] = None
    stats_window_size: int = 100
    tensorboard_log: Optional[str] = None
    verbose: int = 0
    device: Union[th.device, str] = "auto"
    support_multi_env: bool = False
    monitor_wrapper: bool = True
    seed: Optional[int] = None
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    sde_support: bool = True
    supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None
    use_original_space: bool = True
    warmup_zero_action: bool = True


class OffPolicyResidual(CHEF):

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        algocn: Union[Algo_CN, dict],  # algo config node
        fmcn: FoundationModel_CN = OctoS_CN(),
    ):

        self.algocn = Algo_CN(**algocn) if type(algocn) == dict else algocn
        for k, v in asdict(self.algocn).items():
            setattr(self, k, v)

        super().__init__(policy, env, **asdict(self.algocn))

        instructions = self.env.env_method("get_language_instruction")
        print(f"Instructions: {instructions}")
        self.fmcn = fmcn
        self.fm = build_foundation_model(self.fmcn)
        self.fm.reset(instructions)

        self.rescaler = ActionRescaler(
            strategy=self.fmcn.strategy, residual_scale=self.fmcn.residual_scale
        )

        self.observation_space = copy.deepcopy(self.observation_space)
        self.observation_space["agent_partial-action"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # manually remove simpler-img for RP and buffer
        if "simpler-img" in self.observation_space.spaces:
            del self.observation_space.spaces["simpler-img"]

        act_shape = self.action_space.shape[0] - len(self.fmcn.noact)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(act_shape,), dtype=np.float32
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert (
                    self.env is not None
                ), "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            self.fm,
            self.rescaler,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            if self.use_original_space:
                raise NotImplementedError()

            # Rescale and perform action
            actions = self.rescaler.compute_final_action(actions, self.fm_act)
            new_obs, rewards, dones, infos = env.step(actions)

            # add the partial action to the observation
            image = new_obs["simpler-img"]
            # need to untranspose... not transpose again BCHW -> BHWC
            image = np.transpose(image, (0, 2, 3, 1))

            raw, self.fm_act = self.fm.step(image)
            self.fm_act = self.rescaler.dict2act(self.fm_act)

            new_obs["agent_partial-action"] = self.fm_act
            # need to add this retroactively since the env doesn't know about it
            if self._vec_normalize_env is not None:
                self._vec_normalize_env.old_obs["agent_partial-action"] = self.fm_act
            # else: self._last_original_obs, new_obs_ = self._last_obs, new_obs

            for i, done in enumerate(dones):
                if done and infos[i].get("terminal_observation") is not None:
                    infos[i]["terminal_observation"]["agent_partial-action"] = (
                        self.fm_act[i]
                    )
                    if "simple-img" in infos[i]["terminal_observation"]:
                        del infos[i]["terminal_observation"]["simple-img"]
                    infos[i]["terminal_observation"]["simpler-img"]

            self.img = new_obs["simpler-img"]
            del new_obs["simpler-img"]

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            from improve.wrapper import dict_util as du

            # print(f"actions: {du.apply(actions, lambda x: type(x))}")
            # print(f"new_obs: {du.apply(new_obs, lambda x: type(x))}")
            # print(f"new_obs: {du.apply(new_obs, lambda x: x.shape)}")
            # print(f"last_obs: {du.apply(self._last_obs, lambda x: x.shape)}")
            # print(f"buffer_actions: {buffer_actions}")
            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """

        things = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self.fm_act = np.zeros((self.env.num_envs, 7), dtype=np.float32)
            self._last_obs["agent_partial-action"] = self.fm_act

            self.img = self._last_obs["simpler-img"]
            del self._last_obs["simpler-img"]

        return things
