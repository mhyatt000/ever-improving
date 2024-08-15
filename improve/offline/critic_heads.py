from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, Optional, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array
from jax.typing import ArrayLike
# these can be borrowed from original octo code
from octo.model.components.action_heads import (_check_action_window_size,
                                                continuous_loss, masked_mean)
from octo.model.components.base import TokenGroup
from octo.model.components.diffusion import (cosine_beta_schedule,
                                             create_diffusion_model)
from octo.model.components.tokenizers import BinTokenizer
from octo.model.components.transformer import MAPHead
# these can be borrowed from original octo code
from improve.offline.action_heads import (_check_action_window_size,
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
    obs_horizon: int = 2  # window size
    pred_horizon: int = 4
    chunk_size: int = 5
    predictions: int = 1 # number of critics? TBD
    obs_horizon: int = 2
    pred_horizon: int = 4
    chunk_size: int = 5
    action_dim: int = 7
    embedding_dim: int = 384
    dim: int = 1
    embedding_dim: int = 384
    dim: int = 1 
    max_critic: float = 1.0
    loss_type: str = "mse"

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()

        self.action_embed = nn.Dense(self.embedding_dim)
        self.dense1 = nn.Dense(self.embedding_dim)  # [action_embeds + obs_embeds]
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

        bs = embeddings.shape[0]
        if actions is None:  # during initialization actions is not passed :(
            # [bs, 2,4,7]
            actions = jnp.zeros(
                (bs, self.obs_horizon, self.pred_horizon, self.action_dim)
            )

        actions = rearrange(actions, "b w p a -> b w (p a)")
        act_emb = self.action_embed(actions)  # [bs, 2 , 384]

        # [bs, 2, 768]
        both = jnp.concatenate([embeddings, act_emb], axis=-1)
        output = self.dense1(both)

        mean = self.mean_proj(output)
        mean = rearrange(mean, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.dim)

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
        actions = chunk_actions(actions, self.pred_horizon)
        b, w, p, a = actions.shape

        src = self(transformer_outputs, actions, train=train)
        # _check_action_window_size(src, w, self.pred_horizon)
        src = src.squeeze(-1).mean(-1)

        # chunk the values from the batch
        # window_size = src.shape[1]
        tgt = chunk_actions(values, self.pred_horizon)
        tgt = tgt[:, :w]
        tgt = tgt.squeeze(-1).mean(-1) # chunk level critic not intra-chunk critic
        # (batch, window_size, pred_horizon, action_dim)
        # mean = self(transformer_outputs, train=train)

        # loss, metrics = continuous_loss(
        #     mean, actions_chunked, pad_mask[:, :, None, None], loss_type=self.loss_type
        # )

        loss, metrics = continuous_loss(
            src,
            tgt,
            pad_mask, # pad_mask[:, :, None, None],
            loss_type=self.loss_type,
            # pred_values, values_chunked, pad_mask[:, :, None, None], loss_type=self.loss_type
        )

        # Sum over dimension instead of averaging
        loss = loss * self.dim
        metrics["loss"] = metrics["loss"] * self.dim
        metrics["mse"] = metrics["mse"] * self.dim
        metrics["predicted_val"] = src
        metrics["true_val"] = tgt
        
        return loss, metrics

    # def predict(
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

    
class DiscreteCriticHead(nn.Module, CriticHead):
    """
    A basic action decoding head that predicts discretized actions using the transformer token embeddings.


    self.token_per determines how many tokens are used to represent each action.
        - If "" (an empty string): then a single token is responsible for producing the action logits
            for all dimensions at all future prediction horizons.
        - If "pred_horizon", then we use `self.pred_horizon` tokens, each responsible for producing the action logits
            for all dimensions at the corresponding future prediction horizon.
        - If "action_dim_and_pred_horizon", then we use `self.pred_horizon * self.action_dim` tokens, where
            each token is responsible for the logits for the specific dim and timestep.

    If multi-head attention pooling is used (use_map=True), then the correct number of tokens is automatically
    created, otherwise readout_key must have exactly the right number of tokens.
    """

    readout_key: str
    use_map: bool = False
    predictions: int = 1  # number of critics? 
    pred_horizon: int = 1
    action_dim: int = 7
    chunk_size: int = 5
    vocab_size: int = 256
    max_critic: float = 1.0
    # normalization_type: str = "uniform"
    
    embedding_dim: int = 384
    obs_horizon: int = 2 # window size
    quantiles: int = 200
    
    def setup(self):
        if self.use_map:
            self.map_head = MAPHead(num_readouts=self.n_tokens)

        self.action_embed = nn.Dense(self.obs_horizon * self.embedding_dim)
        
        def create_network():
            return [
                nn.Dense(self.embedding_dim),
                nn.Dense(self.embedding_dim // 2),
                nn.Dense(self.quantiles * self.pred_horizon)
                # # Define fully connected layers
                # nn.Dense(self.embedding_dim),
                # nn.Dense(self.quantiles * self.pred_horizon)  # Project to quantiles for each action
            ]
            
        self.critic, self.target_critic = [create_network() for _ in range(2)]
        
    def _forward(
        self, 
        x: jnp.ndarray, 
        main: bool = True
    ) -> jnp.ndarray:
        if main: 
            network = self.critic
        else:
            network = self.target_critic
            
        for layer in network[:-1]:
            x = layer(x)
            x = nn.relu(x)
            
        x = network[-1](x)
        return x
        # return network(x) 

    def __call__(
        self, 
        transformer_outputs: Dict[str, TokenGroup], 
        actions: ArrayLike = None,
        train: bool = True,
        main: bool = True,
    ) -> jax.Array:
        
        initializing = False
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

        bs = embeddings.shape[0]
        if actions is None:  # during initialization actions is not passed :(
            # [bs, 2,4,7]
            initializing = True
            actions = jnp.zeros(
                (bs, self.obs_horizon, self.pred_horizon, self.action_dim)
            )

        actions = rearrange(actions, "b w p a -> b w (p a)")
        act_emb = self.action_embed(actions)  # [bs, 2 , 384]

        # [bs, 2, 768]
        both = jnp.concatenate([embeddings, act_emb], axis=-1)
        
        if initializing:
            self._forward(both, main=False) # to initialize the target network
        
        quantiles = self._forward(both, main=main)
        quantiles = rearrange(quantiles, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.quantiles)
        
        # mean = jnp.tanh(mean / self.max_critic) * self.max_critic
        return quantiles

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        next_transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        rewards: ArrayLike,
        dones: ArrayLike,
        gamma: float,
        delta: float,
        pad_mask: ArrayLike,
        train: bool = True,
    ):
        """Computes the loss for the discretized action objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        dones = jnp.expand_dims(dones, axis=-1)
        rewards = chunk_actions(rewards, self.pred_horizon)        
        actions = chunk_actions(actions, self.pred_horizon)
        
        pred_quantiles = self(transformer_outputs, actions, train=train)
        tgt_quantiles = self(next_transformer_outputs, actions, train=False, main=False)

        tgt_quantiles = rewards + gamma * (1 - dones) * tgt_quantiles   # MC reward means don't use this
        tau = jnp.linspace(0.0, 1.0, self.quantiles)

        u = tgt_quantiles - pred_quantiles
        loss = quantile_loss(u, tau, delta, pad_mask, self.loss_type)
        
        return loss, {"loss": loss, 
                    "predicted_val": pred_quantiles.mean(-1), 
                    "true_val": tgt_quantiles.mean(-1)}
    
    def update_target_network(
            self, 
            tau=0.005
        ):
        """
        Soft update the target network with the main network's parameters.
        """
        
        for main_layer, target_layer in zip(self.critic, self.target_critic):
            # Update the target layer's parameters
            new_target_layer_params = jax.tree_map(
                lambda p_main, p_target: tau * p_main + (1 - tau) * p_target,
                main_layer.variables['params'],
                target_layer.variables['params']
            )
            
            # target_layer.replace(variables={'params':new_target_layer_params})
            
        #     # new_variables = {**target_layer.variables, 'params': new_target_layer_params}
            target_layer.variables["params"] = new_target_layer_params
        # with jax.checking_leaks():
        # new_target_params = jax.tree_map(
        #     lambda p_main, p_target: tau * p_main + (1 - tau) * p_target,
        #     self.critic.variables['params'],
        #     self.target_critic.variables['params']
        # )

        # # Create a new target critic with the updated parameters
        # self.target_critic.variables['params'] = new_target_params
    
        
    def predict(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
    ) -> jax.Array:
        raise NotImplementedError
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        action_logits = self(transformer_outputs, train=train)[:, -1]

        if argmax:
            action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
            action_tokens = jnp.broadcast_to(
                action_tokens, sample_shape + action_tokens.shape
            )
        else:
            dist = distrax.Categorical(logits=action_logits / temperature)
            action_tokens = dist.sample(seed=rng, sample_shape=sample_shape).astype(
                jnp.int32
            )
        return self.action_tokenizer.decode(action_tokens)


class QRDQNCriticHead(DiscreteCriticHead):
    
    readout_key: str = "readout_value"
    quantiles: int = 200
    loss_type: str = "huber"       
    use_map: bool = False
        
@jax.jit
def huber(u: jnp.ndarray, delta=1.0) -> jnp.ndarray:
    abs_u = jnp.abs(u)
    return jnp.where(abs_u <= delta, 0.5 * jnp.square(u), delta * (abs_u - 0.5 * delta))


# @partial(jax.jit, static_argnums=3)
def quantile_loss(
    u: jnp.ndarray,
    tau: jnp.ndarray,
    delta: float,
    pad_mask: jnp.ndarray,
    loss_type: str,
) -> jnp.ndarray:
    
    if loss_type == "l2":
        element_wise_loss = jnp.square(u)
    elif loss_type == "huber":
        element_wise_loss = huber(u, delta)
    else:
        NotImplementedError

    element_wise_loss *= jax.lax.stop_gradient(jnp.abs(tau - (u < 0).astype(jnp.float32)))
    # batch_loss = element_wise_loss.sum(axis=1).mean(axis=1, keepdims=True)
    loss = element_wise_loss.mean(axis=-1)  # average the quantiles
    loss = loss.mean(axis=-1)  # average the prediction horizon
    loss = masked_mean(loss, pad_mask)
    
    return loss


# TODO add options for
# DiscreteCriiticHead (for QRDQN?)
# DiffusionCriticHead
# class DiscreteActionHead(nn.Module, ActionHead):
#     """
#     A basic action decoding head that predicts discretized actions using the transformer token embeddings.


#     self.token_per determines how many tokens are used to represent each action.
#         - If "" (an empty string): then a single token is responsible for producing the action logits
#             for all dimensions at all future prediction horizons.
#         - If "pred_horizon", then we use `self.pred_horizon` tokens, each responsible for producing the action logits
#             for all dimensions at the corresponding future prediction horizon.
#         - If "action_dim_and_pred_horizon", then we use `self.pred_horizon * self.action_dim` tokens, where
#             each token is responsible for the logits for the specific dim and timestep.

#     If multi-head attention pooling is used (use_map=True), then the correct number of tokens is automatically
#     created, otherwise readout_key must have exactly the right number of tokens.
#     """

#     readout_key: str
#     use_map: bool = False
#     token_per: str = "action_dim_and_pred_horizon"
#     pred_horizon: int = 1
#     action_dim: int = 7
#     vocab_size: int = 256
#     normalization_type: str = "uniform"

#     def setup(self):
#         total_output = self.pred_horizon * self.action_dim * self.vocab_size

#         if self.token_per == "":
#             self.n_tokens = 1
#             self.final_layer_size = total_output
#         elif self.token_per == "pred_horizon":
#             self.n_tokens = self.pred_horizon
#             self.final_layer_size = total_output // self.pred_horizon
#         elif self.token_per == "action_dim_and_pred_horizon":
#             self.n_tokens = self.pred_horizon * self.action_dim
#             self.final_layer_size = self.vocab_size
#         else:
#             raise ValueError(f"Invalid token_per: {self.token_per}")

#         if self.use_map:
#             self.map_head = MAPHead(num_readouts=self.n_tokens)

#         self.vocab_proj = nn.Dense(self.final_layer_size)
#         self.action_tokenizer = BinTokenizer(
#             n_bins=self.vocab_size,
#             bin_type=self.normalization_type,
#         )

#     def __call__(
#         self, transformer_outputs: Dict[str, TokenGroup], train: bool = True
#     ) -> jax.Array:
#         """
#         Returns:
#             logits: array w/ shape (batch_size, window_size, pred_horizon, action_dim, vocab_size)
#         """
#         token_group = transformer_outputs[self.readout_key]
#         assert token_group.tokens.ndim == 4, (
#             f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
#             f"but got shape {token_group.tokens.shape}"
#         )
#         if self.use_map:
#             embeddings = self.map_head(token_group, train=train)
#         else:
#             embeddings = token_group.tokens
#             assert (
#                 embeddings.shape[-2] == self.n_tokens
#             ), f"Discrete action head expects {self.n_tokens} tokens"

#         # Now, embeddings is (batch_size, window_size, n_tokens, embedding_size)
#         batch_size, window_size = embeddings.shape[:2]

#         logits = self.vocab_proj(embeddings)
#         logits = logits.reshape(
#             batch_size, window_size, self.pred_horizon, self.action_dim, self.vocab_size
#         )
#         return logits

#     def loss(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         actions: ArrayLike,
#         pad_mask: ArrayLike,
#         train: bool = True,
#     ):
#         """Computes the loss for the discretized action objective.

#         Args:
#             transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
#                 embedding_size)
#             actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
#             pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

#         Returns:
#             loss: float
#             metrics: dict
#         """
#         # get the logits for all the actions by taking the action tokens of each timestep,
#         # unfolding the pred_horizon dim, and projecting to the vocab size
#         # (batch, window_size, pred_horizon, action_dim, token_embedding_size)
#         action_logits = self(transformer_outputs, train=train)

#         window_size = action_logits.shape[1]
#         _check_action_window_size(actions, window_size, self.pred_horizon)

#         actions_chunked = chunk_actions(actions, self.pred_horizon)
#         actions_chunked = actions_chunked[:, :window_size]

#         loss, metrics = discrete_loss(
#             self.action_tokenizer,
#             action_logits,
#             actions_chunked,
#             pad_mask[:, :, None, None],
#         )

#         # For MSE, sum over action dimension instead of averaging
#         metrics["mse"] = metrics["mse"] * self.action_dim

#         return loss, metrics

#     def predict_action(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         train: bool = True,
#         argmax: bool = False,
#         sample_shape: tuple = (),
#         rng: Optional[PRNGKey] = None,
#         temperature: float = 1.0,
#     ) -> jax.Array:
#         """Convenience methods for predicting actions for the final timestep in the window."""
#         # only get the last timestep in the window
#         action_logits = self(transformer_outputs, train=train)[:, -1]

#         if argmax:
#             action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
#             action_tokens = jnp.broadcast_to(
#                 action_tokens, sample_shape + action_tokens.shape
#             )
#         else:
#             dist = distrax.Categorical(logits=action_logits / temperature)
#             action_tokens = dist.sample(seed=rng, sample_shape=sample_shape).astype(
#                 jnp.int32
#             )
#         return self.action_tokenizer.decode(action_tokens)



# class MSEActionHead(ContinuousActionHead):
#     max_action: float = 5.0
#     loss_type: str = "mse"
#     use_map: bool = True


# class L1ActionHead(ContinuousActionHead):
#     max_action: float = 5.0
#     loss_type: str = "l1"
#     use_map: bool = True


# class TokenPerDimActionHead(DiscreteActionHead):
#     token_per: str = "action_dim_and_pred_horizon"


# class DiffusionActionHead(nn.Module):
#     """Predicts actions uses a diffusion process.

#     Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
#     action is then predicted using a diffusion process conditioned on this embedding. The diffusion model
#     architecture is an MLP with residual connections (see `octo.model.components.diffusion`).

#     You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
#     attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
#     stream.
#     """

#     readout_key: str
#     use_map: bool = False
#     pred_horizon: int = 1
#     action_dim: int = 7
#     max_action: float = 5.0
#     loss_type: str = "mse"

#     # diffusion-specific config with sane defaults
#     time_dim: int = 32
#     num_blocks: int = 3
#     dropout_rate: float = 0.0
#     hidden_dim: int = 256
#     use_layer_norm: bool = True
#     diffusion_steps: int = 20

#     def setup(self):
#         if self.use_map:
#             self.map_head = MAPHead()

#         # create the diffusion model (score network)
#         self.diffusion_model = create_diffusion_model(
#             self.action_dim * self.pred_horizon,
#             time_dim=self.time_dim,
#             num_blocks=self.num_blocks,
#             dropout_rate=self.dropout_rate,
#             hidden_dim=self.hidden_dim,
#             use_layer_norm=self.use_layer_norm,
#         )

#         # create beta schedule
#         self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
#         self.alphas = 1 - self.betas
#         self.alpha_hats = jnp.array(
#             [jnp.prod(self.alphas[: i + 1]) for i in range(self.diffusion_steps)]
#         )

#     def __call__(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         time: Optional[ArrayLike] = None,
#         noisy_actions: Optional[ArrayLike] = None,
#         train: bool = True,
#     ) -> jax.Array:
#         """Performs a single forward pass through the diffusion model."""
#         token_group = transformer_outputs[self.readout_key]
#         assert token_group.tokens.ndim == 4, (
#             f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
#             f"but got shape {token_group.tokens.shape}"
#         )
#         if self.use_map:  # Multi-head attention pooling
#             embeddings = self.map_head(token_group, train=train)[:, :, 0]
#         else:  # mean pooling
#             embeddings = token_group.tokens.mean(axis=-2)
#         # Now, embeddings is (batch_size, window_size, embedding_size)

#         # time and noisy_actions are None during initialization, so we replace them with a dummy array
#         if (time is None or noisy_actions is None) and not self.is_initializing():
#             raise ValueError(
#                 "Must provide time and noisy_actions when calling diffusion action head"
#             )
#         elif self.is_initializing():
#             time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
#             noisy_actions = jnp.zeros(
#                 (*embeddings.shape[:2], self.action_dim * self.pred_horizon),
#                 dtype=jnp.float32,
#             )

#         pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
#         return pred_eps

#     def loss(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         actions: ArrayLike,
#         pad_mask: ArrayLike,
#         train: bool = True,
#     ) -> Tuple[Array, Dict[str, Array]]:
#         """Computes the loss for the diffusion objective.

#         Args:
#             transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
#                 embedding_size)
#             actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
#             pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

#         Returns:
#             loss: float
#             metrics: dict
#         """
#         batch_size, window_size = pad_mask.shape
#         _check_action_window_size(actions, window_size, self.pred_horizon)
#         actions_chunked = chunk_actions(actions, self.pred_horizon)
#         actions_chunked = actions_chunked[:, :window_size]
#         # fold action_dim and pred_horizon into one dimension
#         actions_flat = rearrange(actions_chunked, "b w p a -> b w (p a)")
#         actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

#         # piggy-back on the dropout rng chain for diffusion rng
#         rng = self.make_rng("dropout")
#         time_key, noise_key = jax.random.split(rng)
#         time = jax.random.randint(
#             time_key, (batch_size, window_size, 1), 0, self.diffusion_steps
#         )
#         noise = jax.random.normal(noise_key, actions_flat.shape)

#         alpha_hat = self.alpha_hats[time]
#         alpha_1 = jnp.sqrt(alpha_hat)
#         alpha_2 = jnp.sqrt(1 - alpha_hat)
#         noisy_actions = alpha_1 * actions_flat + alpha_2 * noise

#         pred_eps = self(
#             transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
#         )

#         loss, metrics = continuous_loss(
#             pred_eps, noise, pad_mask[:, :, None], loss_type=self.loss_type
#         )
#         # Sum over action dimension instead of averaging
#         loss = loss * self.action_dim
#         metrics["loss"] = metrics["loss"] * self.action_dim
#         metrics["mse"] = metrics["mse"] * self.action_dim
#         return loss, metrics

#     def predict_action(
#         self,
#         transformer_outputs: Dict[str, TokenGroup],
#         rng: PRNGKey,
#         train: bool = True,
#         *args,
#         sample_shape: tuple = (),
#         **kwargs,
#     ) -> jax.Array:
#         """Convenience methods for predicting actions for the final timestep in the window."""
#         module, variables = self.unbind()

#         def scan_fn(carry, time):
#             current_x, rng = carry
#             input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))

#             eps_pred = module.apply(
#                 variables, transformer_outputs, input_time, current_x, train=train
#             )

#             alpha_1 = 1 / jnp.sqrt(self.alphas[time])
#             alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
#             current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

#             rng, key = jax.random.split(rng)
#             z = jax.random.normal(key, shape=current_x.shape)
#             current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)

#             current_x = jnp.clip(current_x, -self.max_action, self.max_action)

#             return (current_x, rng), ()

#         def sample_actions(rng):
#             rng, key = jax.random.split(rng)
#             batch_size, window_size = transformer_outputs[
#                 self.readout_key
#             ].tokens.shape[:2]

#             (actions_flat, _), () = jax.lax.scan(
#                 scan_fn,
#                 (
#                     jax.random.normal(
#                         key,
#                         (batch_size, window_size, self.pred_horizon * self.action_dim),
#                     ),
#                     rng,
#                 ),
#                 jnp.arange(self.diffusion_steps - 1, -1, -1),
#             )

#             actions = rearrange(
#                 actions_flat,
#                 "b w (p a) -> b w p a",
#                 p=self.pred_horizon,
#                 a=self.action_dim,
#             )
#             # only get the last timestep in the window
#             return actions[:, -1]

#         n_samples = int(np.prod(sample_shape))
#         actions = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
#         actions = actions.reshape(sample_shape + actions.shape[1:])
#         return actions
