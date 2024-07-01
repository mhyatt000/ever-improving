import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict as TD
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential as TDS

from improve.wrapper import dict_util as du

from .base import Algo


class SAC(Algo):

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

    def other(self):


    def loss(self, x):

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        with torch.no_grad():

            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )
            # Compute the next Q values: min over all critics targets
            next_q_values = torch.cat(
                self.critic_target(replay_data.next_observations, next_actions),
                dim=1,
            )
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term

            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(
            replay_data.observations, replay_data.actions
        )

        # Compute critic loss
        critic_loss = 0.5 * sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        assert isinstance(critic_loss, torch.Tensor)  # for type checker
        critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    def entropy_loss(self):
        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(
                self.log_ent_coef * (log_prob + self.target_entropy).detach()
            ).mean()
            ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()



    def actor_loss(self):

        # Compute actor loss
        # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = torch.cat(
            self.critic(replay_data.observations, actions_pi), dim=1
        )
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        if gradient_step % self.target_update_interval == 0:
            polyak_update(
                self.critic.parameters(), self.critic_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps

    def on_step_end(self):
        """ log metrics after each update """

        values = q_values_pi.cpu().detach().numpy()
        self.logger.record("stats/values_mean", np.mean(values))
        self.logger.record("stats/values_std", np.std(values))
        self.logger.record("stats/values_hist", wandb.Histogram(values))

        actions = actions_pi.cpu().detach().numpy()
        self.logger.record("stats/actions/x", wandb.Histogram(actions[:, 0]))
        self.logger.record("stats/actions/y", wandb.Histogram(actions[:, 1]))
        self.logger.record("stats/actions/z", wandb.Histogram(actions[:, 2]))

        if len(actions[0]) > 3:
            self.logger.record("stats/actions/roll", wandb.Histogram(actions[:, 3]))
            self.logger.record("stats/actions/pitch", wandb.Histogram(actions[:, 4]))
            self.logger.record("stats/actions/yaw", wandb.Histogram(actions[:, 5]))
        if len(actions[0]) > 6:
            self.logger.record("stats/actions/gripper", wandb.Histogram(actions[:, 6]))

        norms = np.linalg.norm(actions, axis=-1)
        self.logger.record("stats/action_norm", wandb.Histogram(norms))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
