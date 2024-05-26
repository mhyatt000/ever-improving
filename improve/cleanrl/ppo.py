# modified on may 22 2024
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import os.path as osp
import random
import time
import warnings
from dataclasses import asdict

import flax
import flax.linen as nn
import gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import improve
from improve.cleanrl import utils
from improve.cleanrl.net import ActorCritic
from improve.cleanrl.utils import AgentParams, EpisodeStatistics, Storage
from improve.wrappers import residualrl as rrl

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["JAX_TRACEBACK_FILTERING"]="off"

# cuts down on compile time
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")

# NOTE: these caused problems ... turning them off for now
# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
# os.environ["TF_XLA_FLAGS"] = ( "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions")
# os.environ["TF_CUDNN DETERMINISTIC"] = "1"

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

ROOT = osp.dirname(osp.dirname(improve.__file__))
CFG = osp.join(ROOT, "config")


@hydra.main(config_path=CFG, config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size
    cfg.exp.run_name = (
        f"{cfg.exp.env_id}__{cfg.exp.exp_name}__{cfg.exp.seed}__{int(time.time())}"
    )

    if cfg.exp.wandb:
        import wandb

        wandb.init(
            project=cfg.exp.wandb_project_name,
            entity=cfg.exp.wandb_entity,
            sync_tensorboard=True,
            config=vars(cfg),
            name=cfg.exp.run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{cfg.exp.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    key = jax.random.PRNGKey(cfg.exp.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    env = rrl.make(None, "octo-base", None)

    EPSTAT = EpisodeStatistics(
        episode_returns=jnp.zeros(cfg.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(cfg.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(cfg.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(cfg.num_envs, dtype=jnp.int32),
    )

    def step_env(action, EPSTAT):
        obs, reward, terminated, truncated, info = env.step(np.array(action))

        returns = EPSTAT.episode_returns + reward
        length = EPSTAT.episode_lengths + 1

        scale = lambda x: x * (1 - terminated) * (1 - truncated)
        scale_jnp = lambda x: jnp.where(terminated + truncated, returns, x)

        EPSTAT = EPSTAT.replace(
            episode_returns=scale(returns),
            episode_lengths=scale(length),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=scale_jnp(EPSTAT.returned_episode_returns),
            returned_episode_lengths=scale_jnp(EPSTAT.returned_episode_lengths),
        )
        return obs, reward, terminated, truncated, info

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (cfg.num_minibatches * cfg.update_epochs) gradient updates
        frac = (
            1.0
            - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_iterations
        )
        return cfg.optim.learning_rate * frac

    network = ActorCritic(action_dim=env.action_space.shape[0], continuous=True)

    # TODO the residual policy should have its own obs space
    # residual_space = img and partial_action (maybe lang but not for now)
    make_sample = lambda: env.observation_space.sample()

    def obs2robs(obs):
        """convert the obs dict to a residual obs
        usually not needed since env.step returns the residual obs internally
        """
        img = jnp.expand_dims(env.get_image(obs), axis=0)
        action = jnp.expand_dims(
            jnp.reshape(obs["agent"]["partial_action"], (-1)), axis=0
        )
        return (img, action)

    sample = obs2robs(make_sample())

    params = network.init(network_key, sample)


    tx = optax.chain(
        optax.clip_by_global_norm(cfg.optim.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            # TODO use optax lr schedule
            learning_rate=(
                linear_schedule if cfg.optim.anneal_lr else cfg.optim.learning_rate
            ),
            # TODO add weight decay to config cfg
            eps=1e-5,
        ),
    )

    # TODO do you need to jit the functions here? or can you decorate them
    agent_state = TrainState.create(apply_fn=None, params=params, tx=tx)

    # TODO why does this make error
    # network.apply = jax.jit(network.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((cfg.num_steps,) + env.image_space.shape),
        partial=jnp.zeros((cfg.num_steps,) + env.action_space.shape),
        actions=jnp.zeros((cfg.num_steps,) + env.action_space.shape, dtype=jnp.int32),
        logprobs=jnp.zeros(cfg.num_steps),
        dones=jnp.zeros(cfg.num_steps),
        values=jnp.zeros(cfg.num_steps),
        advantages=jnp.zeros(cfg.num_steps),
        returns=jnp.zeros(cfg.num_steps),
        rewards=jnp.zeros(cfg.num_steps),
    )

    # @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        obs: np.ndarray,
        done: np.ndarray,
        storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""

        pi,v = network.apply(agent_state.params, obs)

        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)

        partial = obs[1]
        obs = obs[0]

        storage.obs.at[step].set(obs.squeeze())
        storage.partial.at[step].set(partial.squeeze())
        storage.actions.at[step].set(action.squeeze())
        storage.logprobs.at[step].set(logprob.squeeze())
        storage.dones.at[step].set(done.squeeze())
        storage.values.at[step].set(v.squeeze())

        """ seems to throw errors... :(
        storage = storage.replace(
            # obs contains the partial action too
            obs=storage.obs.at[step].set( obs.squeeze()),  
            partial=storage.partial.at[step].set(partial.squeeze()),
            dones=storage.dones.at[step].set(done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
        )
        """
        return storage, action, key

    @jax.jit
    def get_action_and_value2(params, x, action):
        """ calculates logprobability entropy and value
        occurs during gradient update
        compared with logprob of old weights to measure divergence from old policy

        * applies the network to the current state `x` to obtain hidden representation.
        * applies the actor to obtain the logits of the action probabilities.
        * log probability of the supplied `action` is calculated using logits.

        * logits are normalized and clipped to prevent numerical instability.

        * critic is applied to obtain the value of the current state.
        """

        pi,v = network.apply(params, x)
        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        print("entropy", entropy)
        quit()
        #
        # how to switch to continuous actions
        #


        hidden = network.apply(params["network"], x)
        logits = actor.apply(params["actor"], hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]

        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)

        # entropy of distribution is negative sum of p*log(p) of all actions
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)

        value = critic.apply(params["critic"], hidden).squeeze()
        return logprob, entropy, value

    @jax.jit
    def compute_gae(agent_state, obs, done, storage):
        """Generalized Advantage Estimation (GAE) for PPO

        Defined in https://arxiv.org/abs/1506.02438
        Defined as A^hat_t = sum ( [ gamma * lambda ]^l * delta_{t+l} )
            where delta_{t+l} = r_{t+l} + gamma * V(s_{t+l}) - V(s_t)
                * TD error at time t
                * TD error is the difference between
                the value of the current state and the sum of the reward and the value of the next state
            * gamma is the discount factor
            * lambda is the GAE parameter
            * V(s_t) is the value of the current state

            where A^hat_t is the estimated advantage at time t
            where r_{t+l} is the reward at time t+l
            where V(s_{t+l}) is the value of the state at time t+l
            where l is the lookahead distance
        """

        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))

        hidden = network.apply(agent_state.params["network"], obs)
        value = critic.apply(agent_state.params["critic"], hidden).squeeze()

        lastgaelam = 0
        for t in reversed(range(cfg.num_steps)):
            if t == cfg.num_steps - 1:
                nonterminal = 1.0 - done
                values = value
            else:
                nonterminal = 1.0 - storage.dones[t + 1]
                values = storage.values[t + 1]

            delta = (
                storage.rewards[t]
                + cfg.algo.gamma * values * nonterminal
                - storage.values[t]
            )
            lastgaelam = (
                delta + cfg.algo.gamma * cfg.algo.gae_lambda * nonterminal * lastgaelam
            )

            #  storage.advantages=storage.advantages.at[t].set(lastgaelam.)
            storage = storage.replace(
                advantages=storage.advantages.at[t].set(lastgaelam.squeeze())
            )
        storage = storage.replace(returns=storage.advantages + storage.values)
        return storage

    @jax.jit
    def update_ppo(agent_state: TrainState, storage, key: jax.random.PRNGKey):
        # TODO this is a bit of a mess
        # find a better way to manage multimodal batches
        b_obs = storage.obs.reshape((-1,) + env.image_space.shape)
        b_act = storage.partial.reshape((-1,) + env.action_space.shape)

        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape((-1,) + env.action_space.shape)
        b_advantages = storage.advantages.reshape(-1)
        b_returns = storage.returns.reshape(-1)

        def ppo_loss(params, x, a, logp, advs, returns):
            """PPO surrogate loss function
            has several components:
            * policy loss: encourages actor to take actions that maximize the advantage
                * pg1: TBD
                * pg2: TBD
            * value loss: encourages critic to predict the correct value
            * entropy loss: encourages exploration

            loss = policy_loss - entropy_loss * entropy + value_loss * vf_coef
            """

            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)

            # KL divergence between the old and new policy
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if cfg.algo.norm_adv:  # normalizing advantages can stabilize training
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # Policy loss
            # see PPO paper https://arxiv.org/abs/1707.06347
            # sec.3 eg.7
            pg_loss1 = -advs * ratio
            pg_loss2 = -advs * jnp.clip(
                ratio, 1 - cfg.algo.clip_coef, 1 + cfg.algo.clip_coef
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            # MSE between predicted and observed returns
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

            # Entropy loss
            # encourages exploration
            entropy_loss = entropy.mean()

            loss = (
                pg_loss - cfg.algo.ent_coef * entropy_loss + v_loss * cfg.algo.vf_coef
            )
            parts = pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl)
            return loss, parts  # so that we can view losses separately

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        for _ in range(cfg.update_epochs):
            key, subkey = jax.random.split(key)
            b_inds = jax.random.permutation(subkey, cfg.batch_size, independent=True)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = (
                    ppo_loss_grad_fn(
                        agent_state.params,
                        (b_obs[mb_inds], b_act[mb_inds]),  # used to be b_obs[mb_inds]
                        b_actions[mb_inds],
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        b_returns[mb_inds],
                    )
                )
                agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs, info = env.reset()
    done = np.zeros(cfg.num_envs)

    # env & OctoInference can not be jitted
    # @jax.jit
    def rollout(agent_state, EPSTAT, obs, done, storage, key, global_step):
        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            storage, action, key = get_action_and_value(
                agent_state, obs, done, storage, step, key
            )

            # TRY NOT TO MODIFY: execute the game and log data.
            obs, reward, terminated, truncated, info = step_env(action, EPSTAT)
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return (agent_state, EPSTAT, obs, done, storage, key, global_step)

    """
    MAIN TRAINING LOOP
    """
    for iteration in range(1, int(cfg.num_iterations) + 1):
        iteration_time_start = time.time()
        (agent_state, EPSTAT, obs, done, storage, key, global_step) = rollout(
            agent_state, EPSTAT, obs, done, storage, key, global_step
        )

        storage = compute_gae(agent_state, obs, done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state, storage, key
        )

        avg_episodic_return = np.mean(jax.device_get(EPSTAT.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/avg_episodic_return", avg_episodic_return, global_step
        )
        writer.add_scalar(
            "charts/avg_episodic_length",
            np.mean(jax.device_get(EPSTAT.returned_episode_lengths)),
            global_step,
        )
        writer.add_scalar(
            "charts/learning_rate",
            agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar(
            "charts/SPS_update",
            int(cfg.num_envs * cfg.num_steps / (time.time() - iteration_time_start)),
            global_step,
        )

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
