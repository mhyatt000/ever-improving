from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal, zeros_init
from octo.model.components.vit_encoders import SmallStem, SmallStem16


class Network(nn.Module):

    # not needed see below
    # action_dim: int

    @nn.compact
    def __call__(self, x):
        """
        hard coded input observations  for now
        expects one image and one action in a tuple

        in the future you could use a cfg to prepare tokenizers
            to parse the obs dict for the correct inputs
        """

        img, action = x
        batch = img.shape[0]

        stem = SmallStem(
            use_film=False,
            patch_size=32,
            kernel_sizes=(3, 3, 3, 3),
            strides=(2, 2, 2, 2),
            features=(32, 96, 192, 384),
            padding=(1, 1, 1, 1),
            num_features=512,
            img_norm_type="default",
        )

        x = stem(img)
        x = x.reshape((batch, -1))
        action = action.reshape((batch, -1))

        x = jnp.concatenate([x, action], axis=-1)
        x = nn.Dense(512, kernel_init=zeros_init(), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        # dont need to output in the action space
        # Network creates shared hidden representation for actor and critic heads
        # x = nn.Dense(self.action_dim, kernel_init=zeros_init(), bias_init=constant(0.0))(x)
        # x = nn.tanh(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            self.action_dim, kernel_init=zeros_init(), bias_init=constant(0.0)
        )(x)
