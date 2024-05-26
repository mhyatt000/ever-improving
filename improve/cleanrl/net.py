from typing import Sequence

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal, zeros_init
from octo.model.components.vit_encoders import SmallStem, SmallStem16


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    continuous: bool = True

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

        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.relu(x)

        # if self.continuous:
        mean = nn.Dense(
            self.action_dim, kernel_init=zeros_init(), bias_init=constant(0.0)
        )(x)
        std = nn.Dense(
            self.action_dim, kernel_init=zeros_init(), bias_init=constant(0.0)
        )(x)
        # TODO why apply exp to std
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(std))

        # else:
        # raise NotImplementedError()
        # pi = distrax.Categorical(logits)

        value = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)

        return pi, jnp.squeeze(value, axis=-1)
