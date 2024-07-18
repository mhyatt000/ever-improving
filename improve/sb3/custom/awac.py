import os.path as osp
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import (Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union)

import numpy as np
import torch as th
from gymnasium import spaces
from omegaconf import OmegaConf as OC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   RolloutReturn, Schedule,
                                                   TrainFreq,
                                                   TrainFrequencyUnit)
from stable_baselines3.common.utils import (get_parameters_by_name,
                                            polyak_update)
from stable_baselines3.sac.policies import (Actor, CnnPolicy, MlpPolicy,
                                            MultiInputPolicy, SACPolicy)
from stable_baselines3.sac.sac import SAC
from torch.nn import functional as F

import improve.wrapper.dict_util as du
import wandb
from improve import cn
from improve.data.awac import ep2step, find_tarballs, mk_dataset


def steps2batch(dataset):

    def _mk_batch(steps):
        obs = du.stack([s[0] for s in steps], force=True)
        next_obs = du.stack([s[1] for s in steps], force=True)
        actions = np.array([s[2] for s in steps])
        rewards = np.array([s[3] for s in steps])
        dones = np.array([s[4] for s in steps])
        infos = [s[5] for s in steps]  # i guess these arent stacked?
        return obs, next_obs, actions, rewards, dones, infos

    queue = []
    for sample in dataset:
        steps = list(ep2step(sample))
        for s in steps:
            queue.append(s)
            if len(queue) == 8:
                batch = _mk_batch(queue)
                queue = []
                yield batch


class AWAC(SAC):
    """
    Advantage Weighted Actor Critic (AWAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to AWAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        algocn: cn.AWAC,
    ):
        self.algocn = algocn

        kwargs = deepcopy(algocn)
        for k, v in kwargs.items():
            setattr(self, k, v)
        keys = "buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, policy_kwargs, stats_window_size, tensorboard_log, verbose, device, seed, use_sde, sde_sample_freq, use_sde_at_warmup"
        kwargs = {k: v for k, v in kwargs.items() if k in keys.split(", ")}
        super().__init__(policy, env, **kwargs)

        # SAC will set up the model
        assert self.dataset is not None
        self.dataset = [osp.join(self.log_path, d) for d in self.dataset]
        fnames = list(find_tarballs(self.dataset))
        # fnames = [x for x in list(find_tarballs(self.dataset)) if "eval" in x]
        dataset = mk_dataset(fnames)
        for batch in steps2batch(dataset):
            # print(batch)
            self.replay_buffer.add(*batch)

        """
        for u in [model.actor.mu, model.actor.log_std]:
            u.weight.data.fill_(0)
            u.bias.data.fill_(0)

        self.actor.log_std.weight.data.fill_(0.1)
        self.actor.log_std.bias.data.fill_(1)
        """

    def update(batch):
        """
        Update the actor using the AWAC algorithm.
        - critics give v = q(n actor actions).mean().min()
        - q is q(action from replay).min()
        - advantage is a = q-v
            - advantage of replay action over possible actions from actor
        - actor loss is -softmax(a / beta) * log_prob(actor action)
        """

        v1, v2 = get_value(key, actor, critic, batch, self.num_samples)
        v = jnp.minimum(v1, v2)

        def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
            dist = actor.apply_fn({"params": actor_params}, batch.observations)
            lim = 1 - 1e-5
            actions = jnp.clip(batch.actions, -lim, lim)
            log_probs = dist.log_prob(actions)

            q1, q2 = critic(batch.observations, actions)
            q = jnp.minimum(q1, q2)
            a = q - v

            # we could have used exp(a / beta) here but
            # exp(a / beta) is unbiased but high variance,
            # softmax(a / beta) is biased but lower variance.
            # sum() instead of mean(), because it should be multiplied by batch size.
            actor_loss = -(jax.nn.softmax(a / self.beta) * log_probs).sum()

            return actor_loss, {"actor_loss": actor_loss}

        new_actor, info = actor.apply_gradient(actor_loss_fn)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        # if self.ent_coef_optimizer is not None: optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        # ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            """
            # entropy opt was here
            """

            #
            # compute critic loss
            #

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # td error but NO entropy term
                target_q_values = (
                    replay.rewards + (1 - replay.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay.observations, replay.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            #
            # Compute actor loss
            #

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay.observations)

            # re: actually maybe not?
            # should be deterministic because we are measuring prob of buffer
            # actions_pi = self.actor(replay.observations, deterministic=True)
            # log_prob = log_prob.reshape(-1, 1)

            dist = self.actor.action_dist

            with th.no_grad():
                actions = th.stack(
                    [dist.sample() for _ in range(self.num_samples)], dim=0
                )
                v = th.stack(
                    [
                        th.cat(self.critic(replay.observations, actions[i]), dim=1)
                        for i in range(self.num_samples)
                    ]
                )

                # values is (SAMPLES x B x CRITICS)
                v, _ = v.mean(dim=0).min(dim=1)
                q, _ = th.cat(
                    self.critic(replay.observations, replay.actions), dim=1
                ).min(dim=1)
                advantage = q - v

            # this is the log prob of the replay.actions given actor(obs)
            # clip actions because otherwise log_prob will be nan
            log_prob = dist.log_prob(th.clip(replay.actions, -1, 1)).reshape(-1, 1)
            # log_prob = th.where(th.isnan(log_prob), 0, log_prob)

            mse_loss = F.mse_loss(actions_pi, replay.actions, reduction="none").mean(1)

            # self.logger.record("train/mse_loss", mse_loss.item())
            # actor_losses.append(mse_loss.item())
            # actor_loss +=  mse_loss

            # awac_loss = -(F.softmax(advantage / self.beta) * log_prob).mean()
            awac_loss = -(th.exp(advantage / self.beta) * log_prob).mean()
            # awac_loss = (th.exp(advantage / self.beta) * mse_loss).mean()
            actor_losses.append(awac_loss.item())
            self.logger.record("train/awac_loss", awac_loss.item())
            actor_loss = awac_loss

            gripper_loss = F.smooth_l1_loss(
                actions_pi, replay.actions, reduction="none"
            )[:, -1]
            open = gripper_loss[replay.actions[:, -1] == 1.0].mean()
            close = gripper_loss[replay.actions[:, -1] == -1.0].mean()
            gripper_loss = open + close
            actor_loss += self.gripper_loss_weight * gripper_loss
            self.logger.record("train/gripper_loss", gripper_loss.item())

            log_prob = log_prob.cpu().detach().numpy()
            # log_prob = log_prob[~np.isnan(log_prob)]  # drop nan
            self.logger.record("stats/logp", wandb.Histogram(log_prob))

            # buffer actions
            act = replay.actions.cpu().detach().numpy()
            self.logger.record("stats/replay/x", wandb.Histogram(act[:, 0]))
            self.logger.record("stats/replay/y", wandb.Histogram(act[:, 1]))
            self.logger.record("stats/replay/z", wandb.Histogram(act[:, 2]))
            self.logger.record("stats/replay/yaw", wandb.Histogram(act[:, 3]))
            self.logger.record("stats/replay/pitch", wandb.Histogram(act[:, 4]))
            self.logger.record("stats/replay/roll", wandb.Histogram(act[:, 5]))
            self.logger.record("stats/replay/gripper", wandb.Histogram(act[:, 6]))

            # model actions
            act = actions_pi.cpu().detach().numpy()
            self.logger.record("stats/prediction/x", wandb.Histogram(act[:, 0]))
            self.logger.record("stats/prediction/y", wandb.Histogram(act[:, 1]))
            self.logger.record("stats/prediction/z", wandb.Histogram(act[:, 2]))
            self.logger.record("stats/prediction/yaw", wandb.Histogram(act[:, 3]))
            self.logger.record("stats/prediction/pitch", wandb.Histogram(act[:, 4]))
            self.logger.record("stats/prediction/roll", wandb.Histogram(act[:, 5]))
            self.logger.record("stats/prediction/gripper", wandb.Histogram(act[:, 6]))

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.logger.record("stats/weights/mu", self.actor.mu.weight.mean().item())

            weight = self.actor.mu.weight.cpu().detach().numpy()
            bias = self.actor.mu.bias.cpu().detach().numpy()
            self.logger.record("stats/weights/mu", wandb.Histogram(weight))
            self.logger.record("stats/weights/mu_bias", wandb.Histogram(bias))
            weight = self.actor.log_std.weight.cpu().detach().numpy()
            bias = self.actor.log_std.bias.cpu().detach().numpy()
            self.logger.record("stats/weights/log_std", wandb.Histogram(weight))
            self.logger.record("stats/weights/log_std_bias", wandb.Histogram(bias))

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/ent_coef", np.mean(ent_coefs))

        advantage = advantage.cpu().detach().numpy()
        self.logger.record("stats/advantage", wandb.Histogram(advantage))

        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        # if len(ent_coef_losses) > 0:
        # self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        desc = "You must set the environment before calling learn()"
        assert self.env is not None, desc
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        # offline pretraining
        for i in range(self.offline_steps):
            self.train(batch_size=self.batch_size, gradient_steps=1)
            self.num_timesteps += 1
            if self.num_timesteps % 250 == 0:
                self.logger.dump(step=self.num_timesteps)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(
                        batch_size=self.batch_size, gradient_steps=gradient_steps
                    )

        callback.on_training_end()

        return self
