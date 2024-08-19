from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import lorax
import numpy as np
from einops import rearrange
from jax import Array
from jax.typing import ArrayLike
from octo.model.components.action_heads import (_check_action_window_size,
                                                chunk_actions)


def advantage_loss(
    advantage: jnp.ndarray, objective: jnp.ndarray, beta: float, dist_fn=jax.nn.softmax
):
    """abstract variant of AWAC loss
    policy loss is -softmax(advantage / beta) * objective
    in the original AWAC, objective is log_prob but could also be -mse
    """

    # we could have used exp(a / beta) here but
    # exp(a / beta) is unbiased but high variance,
    # softmax(a / beta) is biased but lower variance.
    # sum() instead of mean(), because it should be multiplied by batch size.
    actor_loss = -(dist_fn(advantage / beta) * objective).sum()

    return actor_loss


from functools import partial


def mk_octo_adv_loss(model, beta):
    return partial(octo_adv_loss_fn, model=model, beta=beta)


@lorax.lora
def octo_adv_loss_fn(params, batch, rng, train, model, beta, update_step, dist_fn=jnp.exp):
    bound = model.module.bind({"params": params}, rngs={"dropout": rng})
    
    embeds = bound.octo_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["pad_mask"],
        train=train,
    )

    # call action loss first to init the diffusion model ?
    action_loss, action_metrics = diffusion_loss(
        bound.heads["action"],
        embeds,  # Action head knows to pull out the action readout_key
        batch["action"],
        pad_mask=batch["observation"]["pad_mask"],
        train=train,
    )

    candidates = predict_actions(
        bound.heads["action"],
        embeds,
        rng=rng,
        train=False,
        sample_shape=(3),
    )

    # final = candidates[-1]  # or something like this

    def toval(x):
        return bound.heads["value"](embeds, x, train=False)
        # return bound.heads["value"](embeds, x, train=False)[0]

    candidates = rearrange(candidates, "s b d w p a -> (s d) b w p a")

    values = jax.vmap(toval)(candidates)  # (60, 64, 2, 4, 1)
    values = values.mean(-1, keepdims=True) # (60, 64, 2, 4, 1)
    values = values.squeeze(-1).mean(-1)[:, :, -1]  # (60,64)

    chunked = chunk_actions(batch["action"], bound.heads["action"].pred_horizon)
    q = bound.heads["value"](embeds, chunked, train=False)
    q = q.mean(-1, keepdims=True)
    q = q.squeeze(-1).mean(-1)[:, -1]  # (64)

    action_metrics["q"] = q.mean()
    action_metrics["value"] = values.mean()

    # final values are of least noisy candidates (q) others are candidates for (v)
    # a = q - v
    a = q - values.mean(0)  # (s d) b
    a = jax.lax.stop_gradient(a)  # stop gradient for critic during actor update

    # not reduced by mean() because it should be scaled by advantage
    action_loss = action_loss.mean(-1)[:, -1]  # no reduce means (bs,w,(p,a))
    action_loss = advantage_loss(a, -action_loss, beta, dist_fn=dist_fn)
    action_metrics["advantage"] = a.mean()
    action_metrics["awac"] = action_loss
    # reduced by advantage_loss with sum since softmax advantages sum to 1

    ### TODO: for QRDQN critic
    next_embeds = bound.octo_transformer(
        batch["next_observation"],
        batch["task"],
        batch["next_observation"]["pad_mask"],
        train=False,
    )
    
    value_loss, value_metrics = bound.heads["value"].loss(
        embeds,
        next_embeds,
        actions=batch["action"],
        rewards=batch["value"],
        dones=jnp.expand_dims(batch["done"], axis=-1),
        gamma=0.99,
        delta=1.0,
        pad_mask=batch["observation"]["pad_mask"],
        train=train,
    )
    
    jax.lax.cond(update_step % 100 == 0, bound.heads["value"].update_target_network, lambda _: None, 0.005)
    
    # value_loss, value_metrics = bound.heads["value"].loss(
    #     embeds,
    #     actions=batch["action"],
    #     values=batch["value"],
    #     pad_mask=batch["observation"]["pad_mask"],
    #     train=train,
    # )

    loss = action_loss + value_loss
    metrics = {"action": action_metrics, "value": value_metrics}
    return loss, metrics


def mk_model_step(model, state):

    @lorax.lora
    def _model_step(params, batch, rng, train=False):
        """for evaluation in env"""
        # use the params and rng from the state
        bound = model.module.bind({"params": params}, rngs={"dropout": state.rng})

        embeds = bound.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )

        candidates = predict_actions(
            bound.heads["action"],
            embeds,
            rng=state.rng,
            train=False,
            sample_shape=(5),
        )

        def toval(x):
            return bound.heads["value"](embeds, x, train=False)

        candidates = rearrange(candidates, "s b d w p a -> (s d) b w p a")

        values = jax.vmap(toval)(candidates)  # (60, 64, 2, 4, 1)
        values = values.squeeze(-1).mean(-1)[:, :, -1]  # (60,64)

        candidates = candidates[:, :, -1]  # (s d) b p a
        actions = candidates[values.argmax(0)]

        return actions

    return _model_step


def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    loss = x * mask

    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    mask = jnp.broadcast_to(mask, loss.shape)
    loss = loss * mask

    mse = jnp.square(pred_value - ground_truth_value)
    mse = mse * mask
    return loss, {"loss": loss.mean(), "mse": mse.mean()}


def diffusion_loss(
    head,
    transformer_outputs,  # : Dict[str, TokenGroup],
    actions,  # : ArrayLike,
    pad_mask,  # : ArrayLike,
    train: bool = True,
):
    """Computes the loss for the diffusion objective.

    Args:
        transformer_ouputs: must contain head.readout_key with shape (batch_size, window_size, num_tokens,
            embedding_size)
        actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
        pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

    Returns:
        loss: float
        metrics: dict
    """
    batch_size, window_size = pad_mask.shape
    _check_action_window_size(actions, window_size, head.pred_horizon)
    actions_chunked = chunk_actions(actions, head.pred_horizon)
    actions_chunked = actions_chunked[:, :window_size]
    # fold action_dim and pred_horizon into one dimension
    actions_flat = rearrange(actions_chunked, "b w p a -> b w (p a)")
    actions_flat = jnp.clip(actions_flat, -head.max_action, head.max_action)

    # piggy-back on the dropout rng chain for diffusion rng
    rng = head.make_rng("dropout")
    time_key, noise_key = jax.random.split(rng)
    time = jax.random.randint(
        time_key, (batch_size, window_size, 1), 0, head.diffusion_steps
    )
    noise = jax.random.normal(noise_key, actions_flat.shape)

    alpha_hat = head.alpha_hats[time]
    alpha_1 = jnp.sqrt(alpha_hat)
    alpha_2 = jnp.sqrt(1 - alpha_hat)
    noisy_actions = alpha_1 * actions_flat + alpha_2 * noise

    pred_eps = head(
        transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
    )

    loss, metrics = continuous_loss(
        pred_eps, noise, pad_mask[:, :, None], loss_type=head.loss_type
    )
    # Sum over action dimension instead of averaging
    loss = loss * head.action_dim
    metrics["loss"] = metrics["loss"] * head.action_dim
    metrics["mse"] = metrics["mse"] * head.action_dim
    
    mean = rearrange(
        pred_eps, "b w (p a) -> b w p a", p=head.pred_horizon, a=head.action_dim
    ) 
    metrics["true_mean"] = actions_chunked
    return loss, metrics


def predict_actions(
    head,
    transformer_outputs,  # : Dict[str, TokenGroup],
    rng,  # : PRNGKey,
    train: bool = True,
    *args,
    sample_shape: tuple = (),
    **kwargs,
) -> jax.Array:
    """Convenience methods for predicting ALL the action timesteps in the window."""
    module, variables = head.unbind()

    def scan_fn(carry, time):
        current_x, rng = carry
        input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))

        eps_pred = module.apply(
            variables, transformer_outputs, input_time, current_x, train=train
        )

        alpha_1 = 1 / jnp.sqrt(head.alphas[time])
        alpha_2 = (1 - head.alphas[time]) / (jnp.sqrt(1 - head.alpha_hats[time]))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng)
        z = jax.random.normal(key, shape=current_x.shape)
        current_x = current_x + (time > 0) * (jnp.sqrt(head.betas[time]) * z)

        current_x = jnp.clip(current_x, -head.max_action, head.max_action)

        # current_x is not returned as an output; only a carry
        return (current_x, rng), current_x

    def sample_actions(rng):
        rng, key = jax.random.split(rng)
        batch_size, window_size = transformer_outputs[head.readout_key].tokens.shape[:2]

        (actions_flat, _), allact = jax.lax.scan(
            scan_fn,
            (
                jax.random.normal(
                    key,
                    (batch_size, window_size, head.pred_horizon * head.action_dim),
                ),
                rng,
            ),
            jnp.arange(head.diffusion_steps - 1, -1, -1),
        )

        allact = jnp.stack(allact, axis=1)

        # since all diffusion timesteps d are used
        actions = rearrange(
            allact,
            "b d w (p a) -> b d w p a",
            p=head.pred_horizon,
            a=head.action_dim,
        )
        # to only get the last timestep in the window:
        # takes window=-1 from (bs,w,p,a) to (bs,p,a)
        # return actions[:, -1]
        # return actions[:, :, -1]  # last window with all diffusion steps
        return actions  # for training use both windows

    n_samples = int(np.prod(sample_shape))
    actions = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
    return actions
    # TODO why does octo reshape?
    # actions = actions.reshape(sample_shape + actions.shape)
    # return actions
