import json
import sys
import os
import os.path as osp
import warnings
from itertools import cycle
from typing import Any, ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import improve
import matplotlib.pyplot as plt
from wandb import wandb
import numpy as np
import scipy.stats as stats
import torch as th
from torch.optim.lr_scheduler import CosineAnnealingLR
from gymnasium import spaces
from gymnasium.spaces import Box, Dict
from improve.data.flex import *
from improve.pac.qrdqn.policies import (CnnPolicy, MlpPolicy, MultiInputPolicy,
                                        QRDQNPolicy, QuantileNetwork)
from improve.wrapper import dict_util as du
from omegaconf import OmegaConf as OC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (get_linear_fn,
                                            get_parameters_by_name,
                                            polyak_update)
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import improve.pac.gr1.models.vision_transformer as vits


SelfQRDQN = TypeVar("SelfQRDQN", bound="QRDQN")
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "datasets", "simpler")


class QRDQN(OffPolicyAlgorithm):
    """
    Quantile Regression Deep Q-Network (QR-DQN)
    Paper: https://arxiv.org/abs/1710.10044
    Default hyperparameters are taken from the paper and are tuned for Atari games
    (except for the ``learning_starts`` parameter).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping (if None, no clipping)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    quantile_net: QuantileNetwork
    quantile_net_target: QuantileNetwork
    policy: QRDQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[QRDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 5e-5,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs=None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.005,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        max_grad_norm: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs=None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.Adam
            # Proposed in the QR-DQN paper where `batch_size = 32`
            self.policy_kwargs["optimizer_kwargs"] = dict(eps=0.01 / batch_size)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
        self.batch_norm_stats = get_parameters_by_name(self.quantile_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.quantile_net_target, ["running_"]
        )
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(
                self.target_update_interval // self.n_envs, 1
            )

    def _create_aliases(self) -> None:
        self.quantile_net = self.policy.quantile_net
        self.quantile_net_target = self.policy.quantile_net_target
        self.n_quantiles = self.policy.n_quantiles

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(
                self.quantile_net.parameters(),
                self.quantile_net_target.parameters(),
                self.tau,
            )
            # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining
        )
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the quantiles of next observation
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                # Compute the greedy actions which maximize the next Q values
                next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(
                    dim=2, keepdim=True
                )
                # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1)
                next_greedy_actions = next_greedy_actions.expand(
                    batch_size, self.n_quantiles, 1
                )
                # Follow greedy policy: use the one with the highest Q values
                next_quantiles = next_quantiles.gather(
                    dim=2, index=next_greedy_actions
                ).squeeze(dim=2)
                # 1-step TD target
                target_quantiles = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_quantiles
                )

            # Get current quantile estimates
            current_quantiles = self.quantile_net(replay_data.observations)

            # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1).
            actions = (
                replay_data.actions[..., None]
                .long()
                .expand(batch_size, self.n_quantiles, 1)
            )
            # Retrieve the quantiles for the actions from the replay buffer
            current_quantiles = th.gather(
                current_quantiles, dim=2, index=actions
            ).squeeze(dim=2)

            # Compute Quantile Huber loss, summing over a quantile dimension as in the paper.
            loss = quantile_huber_loss(
                current_quantiles, target_quantiles, sum_over_quantiles=True
            )
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(
                observation, state, episode_start, deterministic
            )
        return action, state

    def learn(
        self: SelfQRDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "QRDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfQRDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "quantile_net",
            "quantile_net_target",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


def get_observation(batch):
    batch = {
        "simpler-img": batch["observation"]["simpler-img"].permute(0, 3, 1, 2),
        "agent_partial-action": batch["observation"]["agent_partial-action"],
    }
    return batch


# @jay this function is not needed anymore
# read dict_utils.py
""" 
def shift_batch(batch, padding_value=0):
    shifted_batch = {}
    for key, value in batch.items():
        if isinstance(value, th.Tensor):
            shifted_value = th.roll(value, shifts=1, dims=0)
            shifted_value = shifted_value[1:]
            shifted_batch[key] = shifted_value
        else:
            shifted_batch[key] = shift_batch(value, padding_value)
    return shifted_batch


def remove_first(batch):
    for key, value in batch.items():
        if isinstance(value, th.Tensor):
            batch[key] = value[1:]
        else:
            batch[key] = remove_first(value)
    return batch
"""


def preprocess_batch(batch):
    """Preprocesses batch
    - by removing first and last elements along the time dimension
    - by aligning the current step with the future step
    """

    chop = lambda x: x[1:-1, ...]
    a = du.apply(batch, chop)
    b = du.apply(batch, lambda x: chop(th.roll(x, shifts=-1, dims=0)))
    return a, b


def load_data(batch_size=33):
    dataset = HDF5IterDataset(DATA_DIR, loop=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)
    return cycle(loader)


def initialize_model():
    observation_space = Dict(
        {
            "agent_partial-action": Box(-float("inf"), float("inf"), (7,), np.float32),
            "agent_qpos": Box(-float("inf"), float("inf"), (11,), np.float32),
            "agent_qvel": Box(-float("inf"), float("inf"), (11,), np.float32),
            "simpler-img": Box(low=0, high=255, shape=(3, 69, 91), dtype=np.uint8),
        }
    )

    action_space = Box(
        np.array([-1.0, -1.0, -1.0, -1.5707964, -1.5707964, -1.5707964, -1.0]),
        np.array([1.0, 1.0, 1.0, 1.5707964, 1.5707964, 1.5707964, 1.0]),
        (7,),
        np.float32,
    )

    def lr_schedule(n_steps):
        start_lr = 5e-5
        end_lr = 1e-6
        max_steps = 1e6
        return start_lr - (n_steps * (start_lr - end_lr)) / max_steps

    model = MultiInputPolicy(
        Dict(
            {
                "simpler-img": observation_space["simpler-img"],
                "agent_partial-action": observation_space["agent_partial-action"],
            }
        ),
        action_space,
        lr_schedule,
    ).to('cuda')
    pprint(model)
    return model


def train(model, loader, cfg):
    # @jay we should split the data into train and eval datasets
    # I thing torch or sklearn has a function for that

    ### remove the first 4 batches for eval
    # for _ in range(4):
    # batch = next(loader)
    run = None
    if cfg.logging:
        fun = wandb.init(
            project="residualrl",
            job_type="train",
            name=""
        )

    n_steps = 100_000

    model = initialize_model()
    model = torch.compile(model)
    model.set_training_mode(True)
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)
    losses = []
    
    bar = tqdm(total=n_steps)
    for _ in range(n_steps):
        # Sample replay buffer
        # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

        batch = next(loader)
        
        if batch['reward'].shape[0] < cfg.batch_size:
            continue
        
        batch = du.apply(batch, lambda x: x.to('cuda'))
        current, future = preprocess_batch(batch)
        current_obs = get_observation(current)
        next_obs = get_observation(future)
        
    

        with th.no_grad():
            # Compute the quantiles of future observation
            # next_quantiles = self.quantile_net_target(replay_data.next_observations)

            # get the future quantiles
            next_quantiles, next_greedy_actions = model._predict(next_obs, False)

            next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(
                dim=2, keepdim=True
            )

            # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1)
            next_greedy_actions = next_greedy_actions.expand(cfg.batch_size - 2, 200, 1)

            # Follow greedy policy: use the one with the highest Q values
            next_quantiles = next_quantiles.gather(
                dim=2, index=next_greedy_actions
            ).squeeze(dim=2)

            # breakpoint()
            # 1-step TD target
            target_quantiles = (
                current["reward"].unsqueeze(1)
                + (1 - current["terminated"].long().unsqueeze(1))
                * 0.99
                * next_quantiles
            )

        current_quantiles, _ = model._predict(current_obs, False)

        # grab the first n_quantiles (200)
        current_quantiles = current_quantiles.squeeze(dim=2)

        # Compute Quantile Huber loss, summing over a quantile dimension as in the paper.
        loss = quantile_huber_loss(
            current_quantiles, target_quantiles, sum_over_quantiles=True
        )

        # Optimize the policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()

        # move to cpu and remove from computation graph
        loss = loss.detach().cpu().numpy().item()
        losses.append(loss)
        
        if cfg.logging:
            wandb.log({"train/loss": loss, "train/best_loss": min(losses), "train/learning_rate": scheduler.get_last_lr()[0]})
        
        # f string with scientific notation
        desc = f"loss: {loss:.2e} | best: {min(losses):.2e}"
        bar.set_description(desc)
        bar.update(1)

        # if loss <= min(losses):
            # th.save(model.state_dict(), osp.join(improve.WEIGHTS, f"qrdqn_{loss:.2e}.pth"))

        with open("losses.json", "w") as f:
            json.dump(losses, f)
    
    if run:        
        run.finish()


def main():
    is_training = False
    is_logging = False
    if len(sys.argv) > 1:
        is_training = sys.argv[1] == "train"
        
        if len(sys.argv) > 2:
            is_logging = sys.argv[2] == "log"

    cfg = {
        "batch_size": 256,
        "training": is_training,
        "logging": is_logging
    }
    cfg = OC.create(cfg)
    
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0).to(device)
    # checkpoint = th.load(osp.join(improve.WEIGHTS, 'mae_pretrain_vit_base.pth'))
    # model_mae.load_state_dict(checkpoint["model"], strict=False)
    
    # batch = next(iter(loader))
    # obs = get_observation(batch)
    # output = model_mae(obs["simpler-img"].to(th.float32).to(device))

    ### Training script
    if cfg.training:
        loader = load_data(batch_size=cfg.batch_size)
        model = initialize_model()
        train(model, loader, cfg)

        breakpoint()

    ### Evaluation script
    loader = load_data(batch_size=256)
    models = [x for x in os.listdir(improve.WEIGHTS) if x.startswith("qrdqn")]
    model = th.load(osp.join(improve.WEIGHTS, models[-1]))

    print(f"Model {models[-1]} loaded")

    has_reward = None
    while not has_reward:
        batch = next(loader)
        rewards = batch["reward"]
        has_reward = (rewards != 0).any()

    current_obs = get_observation(batch)

    with th.no_grad():
        current_quantiles, _ = model._predict(current_obs, False)
        current_quantiles = current_quantiles.squeeze(dim=2)
        current_quantiles = current_quantiles.squeeze(0)

    quantiles = F.softmax(current_quantiles, dim=1)
    lims = (quantiles.min().item(), quantiles.max().item())
    space = np.linspace(0, 1, 200)

    for i, q in tqdm(zip(range(len(quantiles)), quantiles), total=len(quantiles)):

        plt.figure()
        plt.scatter(space, q.numpy())

        # vertical line at rewards[i]
        plt.axvline(x=rewards[i].item(), color="r", linestyle="--")

        plt.ylim(lims)
        plt.savefig(f"step_{i}.png")
        plt.close()


if __name__ == "__main__":
    main()
