import flax
import jax.numpy as jnp
from gymnasium.spaces.dict import Dict


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

    def to_dict(self):
        return {
            "network_params": dict(self.network_params),
            "actor_params": dict(self.actor_params),
            "critic_params": dict(self.critic_params),
        }


@flax.struct.dataclass
class Storage:
    # used to be just one array... could be dict of arrays
    # re: use partial to store the partial action for now
    obs: jnp.array
    partial: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

    # TODO add info? ... its not a jnp.array :(


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def traverse(it, func):
    if type(it) in (dict, Dict):
        return {k: traverse(v, func) for k, v in it.items()}
    if isinstance(it, list):
        return [traverse(item, func) for item in it]
    else:
        return func(it)
