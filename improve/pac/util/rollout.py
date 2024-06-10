import torch


class RolloutBuffer:

    def __init__(self, seq, obs_shape, action_shape):
        self.seq = seq # sequence length
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reset()

    def reset(self):
        self.observations = torch.zeros(
            (self.rollout_length, self.num_agents, *self.obs_shape)
        )
        self.actions = torch.zeros(
            (self.rollout_length, self.num_agents, *self.action_shape)
        )
        self.rewards = torch.zeros((self.rollout_length, self.num_agents, 1))
        self.values = torch.zeros((self.rollout_length, self.num_agents, 1))
        self.log_probs = torch.zeros((self.rollout_length, self.num_agents, 1))
        self.dones = torch.zeros((self.rollout_length, self.num_agents, 1))
        self.advantages = torch.zeros((self.rollout_length, self.num_agents, 1))

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for t in reversed(range(self.rollout_length - 1)):
            self.returns[t] = self.returns[t + 1] * \
                gamma * (1 - self.dones[t]) + self.rewards[t]
