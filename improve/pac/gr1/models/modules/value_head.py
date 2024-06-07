from typing import Any, Dict, List, Optional, Type
from gym import spaces
import torch as th
import torch.nn as nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class QuantileNetwork(BasePolicy):
    """
    Quantile network for QR-DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        action_space_n = 1,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.n_quantiles = n_quantiles
        self.action_space_n = action_space_n
        action_dim = int(action_space_n)  # number of actions
        quantile_net = create_mlp(self.features_dim, action_dim * self.n_quantiles, self.net_arch, self.activation_fn)
        self.quantile_net = nn.Sequential(*quantile_net)

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Predict the quantiles.

        :param obs: Observation
        :return: The estimated quantiles for each action.
        """
        # hidden = self.extract_features(obs, self.features_extractor)
        # print(hidden.shape)
        
        # Block -> Flatten
        hidden = obs
        # hidden = hidden.view(hidden.size(0), -1)
        quantiles = self.quantile_net(hidden)
        # print(quantiles.shape)
        
        return quantiles.view(-1, self.n_quantiles, self.action_space_n)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_distributions = self(observation)
        q_values = q_distributions.mean(dim=1)

        # generate random actions
        if deterministic:
            for i in range(2):
                random_observation = {'simpler-img': observation['simpler-img'], 'agent_partial-action': th.rand(32, 7) * 2 - 1}
                random_q_distribution =  self(random_observation)
                # random_q_values = random_q_distribution.mean(dim=1)
                q_distributions = th.cat((q_distributions,random_q_distribution), dim=1)

        # Greedy action
        action = q_values.argmax(dim=1)
        return q_distributions, action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                n_quantiles=self.n_quantiles,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data