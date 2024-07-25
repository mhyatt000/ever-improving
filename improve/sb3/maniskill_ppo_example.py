import os
import os.path as osp
import warnings
from pprint import pprint

import gymnasium as gym
# import mani_skill2.envs
import numpy as np
import simpler_env as simpler
import stable_baselines3 as sb3
from omegaconf import OmegaConf as OC
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from wandb.integration.sb3 import WandbCallback

import hydra
import improve
import improve.hydra.resolver
import wandb
from improve.env import make_env, make_envs
from improve.log.wandb import WandbLogger
from improve.sb3 import util
from improve.sb3.custom import AWAC, PPO, RP_SAC, SAC, TQC
from improve.wrapper import dict_util as du

warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):
    
    if cfg.job.wandb.use:
        print("Using wandb")
        run = wandb.init(
            project="awac",
            dir=cfg.callback.log_path,
            job_type="train",
            # sync_tensorboard=True,
            monitor_gym=True,
            name=cfg.job.wandb.name,
            group=cfg.job.wandb.group,
            config=OC.to_container(cfg, resolve=True),
        )
        wandb.config.update({"name": run.name})

    print(OC.to_yaml(cfg, resolve=True))  # keep after wandb so it logs

    num_envs = cfg.env.n_envs
    max_episode_steps = cfg.env.max_episode_steps
    
    ### CHANGED
    log_dir = osp.join(cfg.callback.log_path, wandb.run.name) if cfg.job.wandb.use else None
    
    rollout_steps = cfg.algo.get("n_steps", None) or 4800

    if cfg.job.seed is not None:
        set_random_seed(cfg.job.seed)

    if cfg.job.wandb.use:
        # initialize wandb logger
        format_strings = ["stdout", "tensorboard"]
        folder = "home/zero-shot/sb3_logs"
        logger = WandbLogger(folder, format_strings)

    print(f"eval_only: {cfg.train.use_train}")
    print(f"fm_name: {cfg.env.foundation.name}")

    eval_only = not cfg.train.use_train
    # create eval environment

    env, eval_env = make_envs(
        cfg,
        log_dir,
        eval_only=eval_only,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
    )
    print(env)

    algo_kwargs = OC.to_container(cfg.algo, resolve=True)
    policy_kwargs = algo_kwargs.get("policy_kwargs", {})
    del algo_kwargs["name"]
    del algo_kwargs["policy_kwargs"]

    # NOTE is this needed? already added in cn.Algo
    policy_kwargs.update(
        {"optimizer_kwargs": {"betas": (0.999, 0.999), "weight_decay": 1e-4}}
    )

    if cfg.algo.name == "sac":
        del algo_kwargs["use_original_space"]

    if cfg.algo.name == "ppo":
        algo_kwargs.update(
            dict(
                verbose=1,
                n_steps=rollout_steps // num_envs,
                batch_size=400,
                gamma=0.8,
                n_epochs=15,
                tensorboard_log=log_dir,
            )
        )

    # Define the policy configuration and algorithm configuration
    algo = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": RP_SAC if cfg.env.fm_loc.value == "central" else SAC,
        "awac": AWAC,
        "rp_sac": RP_SAC,
        "tqc": TQC,
    }[cfg.algo.name]

    from improve import cn

    if cfg.algo.name == "rp_sac":
        algo_kwargs = OC.to_container(cfg.algo, resolve=True)
        algocn = cn.RP_SAC(**algo_kwargs)

        if cfg.env.foundation.name == "octo-small":
            fmcn = cn.OctoS(**OC.to_container(cfg.env.foundation, resolve=True))
        elif cfg.env.foundation.name == "rtx":
            fmcn = cn.RTX(**OC.to_container(cfg.env.foundation, resolve=True))

        model = algo("MultiInputPolicy", env, algocn, fmcn)

    elif cfg.algo.name == "awac":
        algocn = OC.to_container(cfg.algo, resolve=True)
        model = AWAC("MultiInputPolicy", env, algocn, cfg.env.task)

    else:
        model = algo(
            "MultiInputPolicy",
            env,
            **algo_kwargs,
            policy_kwargs=policy_kwargs,
        )

    print(model.policy)

    if cfg.job.wandb.use:
        model.set_logger(logger)

    n_eval = 10

    if eval_only:
        model_path = cfg.job.name
        if model_path is None:
            model_path = osp.join(log_dir, "latest_model")
        # Load the saved model
        model = model.load(model_path)

    else:
        # define callbacks to periodically save our model and evaluate it to help monitor training
        # the below freq values will save every 10 rollouts

        # this might negatively affect training
        post_eval = (
            util.ReZeroAfterFailure(threshold=0.2, verbose=1)
            if cfg.train.use_zero_init
            else None
        )
        # post_eval if cfg.env.foundation.name == 'octo-base' else None
        post_eval = None
        print("WARN: using post eval callback")
        print(type(post_eval))

        eval_callback = EvalCallback(
            eval_env,
            callback_after_eval=post_eval,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=rollout_steps // num_envs,
            deterministic=True,
            render=True,
            n_eval_episodes=max(num_envs, 5),
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10 * rollout_steps // num_envs,
            save_path=log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        callbacks = [checkpoint_callback, eval_callback] if log_dir else [eval_callback]

        if cfg.job.wandb.use:
            wandbCb = WandbCallback(
                gradient_save_freq=rollout_steps // num_envs,
                log="gradients",
                verbose=2,
            )
            callbacks += [wandbCb]

        if cfg.train.use_zero_init:
            util.zero_init(model, cfg.algo.name)

        print("Training model")
        # Train an agent with PPO for args.total_timesteps interactions
        model.learn(
            cfg.train.n_steps,
            callback=callbacks,
            progress_bar=True,
        )
        # Save the final model
        if log_dir:
            model.save(osp.join(log_dir, "latest_model"))

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=True,
        return_episode_rewards=True,
        n_eval_episodes=n_eval,
    )

    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)

    # close all envs
    eval_env.close()
    if not eval_only:
        env.close()

    if cfg.job.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    main()
