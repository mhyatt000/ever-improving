from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array
from jax.typing import ArrayLike
from octo.model.components.base import TokenGroup
from octo.model.components.diffusion import (cosine_beta_schedule,
                                             create_diffusion_model)
from octo.model.components.tokenizers import BinTokenizer
from octo.model.components.transformer import MAPHead
# these can be borrowed from original octo code
from octo.model.components.action_heads import (_check_action_window_size,
                                                     continuous_loss,
                                                     masked_mean)
from octo.utils.typing import PRNGKey


class CriticHead(ABC):
    """Prediction modules that take in the transformer token outputs and predicted actions
    predict action values.

    While action heads use action chunking (ACT), the critic can only evaluate the
    ground-truth actions
    """

    @abstractmethod
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        rewards: ArrayLike,
        pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        train: bool = False,
    ) -> Array:
        """Predict the value for the actions from last timestep in the window.
        Returns shape (*sample_shape, batch_size, pred_horizon, action_dim).
        """
        raise NotImplementedError


def chunk_actions(actions: ArrayLike, pred_horizon: int) -> Array:
    """Chunk actions for predicting actions `pred_horizon` steps into the future.

    The resulting actions have shape (batch, actions.shape[-2] - (pred_horizon - 1), pred_horizon, action_dim)

    For example: chunk_actions([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
        ]

    """
    assert (
        actions.ndim == 3
    ), f"Expected actions to have shape (batch, window_size, action_dim), but got shape {actions.shape}"
    window_size = actions.shape[1]
    assert (
        window_size >= pred_horizon
    ), f"pred_horizon {pred_horizon} too large for window size {window_size}"
    chunk_window_size = window_size - (pred_horizon - 1)

    curr_step = jnp.arange(chunk_window_size)
    action_offset = jnp.arange(pred_horizon)
    chunk_indices = curr_step[:, None] + action_offset[None, :]
    return actions[:, chunk_indices]


def discrete_loss(
    discrete_tokenizer: BinTokenizer,
    logits: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
) -> Array:
    """
    Args:
        discrete_tokenizer: BinTokenizer to use on ground_truth_value
        logits: shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in w/ shape (batch_dims...)
        mask: broadcastable to ground_truth_value
    """
    labels = discrete_tokenizer(ground_truth_value)
    labels_one_hot = jax.nn.one_hot(labels, logits.shape[-1])

    loss = jnp.sum(jax.nn.log_softmax(logits, axis=-1) * labels_one_hot, axis=-1)
    loss = -masked_mean(loss, mask)

    # compute accuracy between predicted actions and target actions
    pred_label = jnp.argmax(logits, axis=-1)
    accuracy = pred_label == labels
    accuracy = masked_mean(accuracy, mask)

    # detokenize the predicted actions
    pred_value = discrete_tokenizer.decode(pred_label)
    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "accuracy": accuracy,
    }


class ContinuousCriticHead(nn.Module, CriticHead):
    """Predicts continuous values (as opposed to discretized).

    Continuous values are predicted by tanh squashing the model output to [-max_critic, max_critic], and then
    optimized using a standard regression loss.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str = "readout_value"
    use_map: bool = False
    predictions: int = 1  # number of critics? TBD
    obs_horizon: int = 2
    pred_horizon: int = 4
    chunk_size: int = 5
    action_dim: int = 7
    embedding_dim: int = 384
    dim: int = 1
    max_critic: float = 1.0
    loss_type: str = "mse"

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()

        self.action_embed = nn.Dense(self.obs_horizon * self.embedding_dim)
        self.dense1 = nn.Dense(2 * self.embedding_dim)  # [action_embeds + obs_embeds]
        self.mean_proj = nn.Dense(self.pred_horizon * self.dim)

    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Returns:
            mean: Predicted values w/ shape (batch_size, window_size, pred_horizon, dim)
        """
        token_group = transformer_outputs[self.readout_key]

        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        if actions is None:
            actions = jnp.zeros(
                (embeddings.shape[0], self.chunk_size, self.action_dim)
            )  # [bs, 4, 7]

        actions = actions.reshape(-1, self.chunk_size * self.action_dim)  # [bs, 28]
        action_embeddings = self.action_embed(actions)  # [bs, 2 * 384]
        action_embeddings = action_embeddings.reshape(
            *embeddings.shape[0:-1], self.embedding_dim
        )  # [bs, 2, 384]

        joined_embeddings = jnp.concatenate(
            [embeddings, action_embeddings], axis=-1
        )  # [bs, 2, 768]
        output = self.dense1(joined_embeddings)  # [bs, 2, 768]

        mean = self.mean_proj(output)  # [bs, 2, 5]
        mean = rearrange(
            mean, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.dim
        )  # [bs, 2, 5, 1]

        mean = jnp.tanh(mean / self.max_critic) * self.max_critic
        return mean

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        values: ArrayLike,  # [63, 5, 1]
        pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        # (batch, window_size, pred_horizon, action_dim)
        # mean = self(transformer_outputs, train=train)

        ### TODO: for MSE critic head, measure MSE btwn pred_values and values and return mean of batch as loss
        pred_values = self(transformer_outputs, actions, train=train)

        # window_size = mean.shape[1]
        window_size = pred_values.shape[1]
        _check_action_window_size(actions, window_size, self.pred_horizon)
        # actions_chunked = chunk_actions(actions, self.pred_horizon)
        # actions_chunked = actions_chunked[:, :window_size]
        values_chunked = chunk_actions(values, self.pred_horizon)
        values_chunked = values_chunked[:, :window_size]

        # loss, metrics = continuous_loss(
        #     mean, actions_chunked, pad_mask[:, :, None, None], loss_type=self.loss_type
        # )

        loss, metrics = continuous_loss(
            pred_values,
            values_chunked,
            pad_mask[:, :, None, None],
            loss_type=self.loss_type,
        )

        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim

        return loss, metrics

    def predict(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        raise NotImplementedError
        # only get the last timestep in the window
        # (batch, pred_horizon, action_dim)
        mean = self(transformer_outputs, train=train)[:, -1]
        return jnp.broadcast_to(mean, sample_shape + mean.shape)


class MSECriticHead(ContinuousCriticHead):
    max_action: float = 5.0
    loss_type: str = "mse"
    use_map: bool = True

# TODO add options for 
# DiscreteCriiticHead (for QRDQN?)
# DiffusionCriticHead
