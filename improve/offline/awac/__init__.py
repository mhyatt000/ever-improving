from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import lorax
import numpy as np
from einops import rearrange


def advantage_loss(advantage: jnp.ndarray, objective: jnp.ndarray, beta: float):
    """abstract variant of AWAC loss
    policy loss is -softmax(advantage / beta) * objective
    in the original AWAC, objective is log_prob but could also be -mse
    """

    # we could have used exp(a / beta) here but
    # exp(a / beta) is unbiased but high variance,
    # softmax(a / beta) is biased but lower variance.
    # sum() instead of mean(), because it should be multiplied by batch size.
    actor_loss = -(jax.nn.softmax(advantage / beta) * objective).sum()

    return actor_loss


from functools import partial


def mk_octo_adv_loss(model, beta):
    return partial(octo_adv_loss_fn, model=model, beta=beta)


@lorax.lora
def octo_adv_loss_fn(params, batch, rng, train, model, beta):
    bound = model.module.bind({"params": params}, rngs={"dropout": rng})

    embeds = bound.octo_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["pad_mask"],
        train=train,
    )

    # call action loss first to init the diffusion model ?
    action_loss, action_metrics = bound.heads["action"].loss(
        embeds,  # Action head knows to pull out the action readout_key
        batch["action"],
        pad_mask=batch["observation"]["pad_mask"],
        train=train,
    )

    candidates = predict_actions(
        bound.heads["action"], embeds, rng=rng, train=False
    )  # sample_shape=(3),
    # final = candidates[-1]  # or something like this
    values = bound.heads["value"](embeds, candidates, train=False)
    q = bound.heads["value"](embeds, batch["action"], train=False)

    # final values are of least noisy candidates (q) others are candidates for (v)
    # a = q - v
    a = q - values[:-1].mean(-1)
    a = jax.lax.stop_gradient(a)  # stop gradient for critic during actor update

    action_loss = advantage_loss(a, action_loss, beta)

    value_loss, value_metrics = bound.heads["value"].loss(
        embeds,
        actions=batch["action"],
        values=batch["value"],
        pad_mask=batch["observation"]["pad_mask"],
        train=train,
    )

    loss = action_loss + value_loss
    metrics = {"action": action_metrics, "value": value_metrics}
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

        return (current_x, rng), ()

    def sample_actions(rng):
        rng, key = jax.random.split(rng)
        batch_size, window_size = transformer_outputs[head.readout_key].tokens.shape[:2]

        (actions_flat, _), () = jax.lax.scan(
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

        actions = rearrange(
            actions_flat,
            "b w (p a) -> b w p a",
            p=head.pred_horizon,
            a=head.action_dim,
        )
        # to only get the last timestep in the window: return actions[:, -1]
        return actions

    n_samples = int(np.prod(sample_shape))
    actions = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
    actions = actions.reshape(sample_shape + actions.shape)
    return actions
