import argparse
import math
import os
import os.path as osp
import warnings
from functools import partial
from pprint import pprint

import gymnasium as gym
import hydra
import mani_skill2.envs
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from mani_skill2.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from stable_baselines3 import A2C, PPO, HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

import improve
import improve.config.prepare
import improve.config.resolver
from improve.sb3 import util
from improve.sb3.util import MyCallback, ReZeroCallback, WandbLogger
from improve.wrapper import residualrl as rrl
from improve.wrapper.normalize import NormalizeObservation, NormalizeReward
from improve.wrapper.probe import ProbeEnv


# Defines a continuous, infinite horizon, task where terminated is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=50,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    if cfg.job.wandb.use:
        print('Using wandb')
        wandb.init(
            project="residualrl-maniskill2demo",
            dir=cfg.callback.log_path,
            job_type="train",
            # sync_tensorboard=True,
            monitor_gym=True,
            name=cfg.job.wandb.name,
            group=cfg.job.wandb.group,
            tags=[t for t in cfg.job.wandb.tags],
            config=OC.to_container(cfg, resolve=True),
        )
    pprint(OC.to_container(cfg, resolve=True))  # keep after wandb so it logs

    # args = parse_args()
    num_envs = cfg.env.n_envs
    max_episode_steps = cfg.env.max_episode_steps
    log_dir = cfg.callback.log_path
    rollout_steps = 4800

    if cfg.job.seed is not None:
        set_random_seed(cfg.job.seed)

    def make_env(cfg, max_episode_steps: int = None, record_dir: str = None):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            import improve.wrapper.residualrl as rrl

            env = rrl.make(
                cfg.env,
                obs_mode='state_dict',
                render_mode="cameras",
                max_episode_steps=max_episode_steps,
                renderer_kwargs={
                    "offscreen_only": True,
                    "device": "cuda:0",
                },
            )

            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
                env = RecordEpisode(env, record_dir, info_on_video=True)
            return env

        return _init

    eval_only = not cfg.train.use_train
    # create eval environment
    if eval_only:
        record_dir = osp.join(log_dir, "videos/eval")
    else:
        record_dir = osp.join(log_dir, "videos")
    eval_env = SubprocVecEnv([make_env(cfg, record_dir=record_dir) for _ in range(1)])
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(cfg.job.seed)
    eval_env.reset()

    if eval_only:
        env = eval_env
    else:
        # Create vectorized environments for training
        env = SubprocVecEnv(
            [
                make_env(cfg, max_episode_steps=max_episode_steps)
                for _ in range(num_envs)
            ]
        )
        env = VecMonitor(env)
        env.seed(cfg.job.seed)
        env.reset()

    algo_kwargs = OC.to_container(cfg.algo, resolve=True)
    policy_kwargs = algo_kwargs.get("policy_kwargs", {})
    del algo_kwargs["name"]
    del algo_kwargs["policy_kwargs"]

    policy_kwargs.update(
        {"optimizer_kwargs": {"betas": (0.999, 0.999), "weight_decay": 1e-4}}
    )

    if cfg.algo.name == "sac":
        del algo_kwargs['use_original_space']
        del algo_kwargs['warmup_zero_action']

    if cfg.algo.name == 'ppo':
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
        "sac": SAC,
    }[cfg.algo.name]
    model = algo(
        "MultiInputPolicy",
        env,
        **algo_kwargs,
        policy_kwargs=policy_kwargs,
    )

    if cfg.job.wandb.use:
        # initialize wandb logger
        format_strings = ["stdout", "tensorboard"]
        folder = "home/zero-shot/sb3_logs"
        model.set_logger(WandbLogger(folder, format_strings))

    if eval_only:
        model_path = cfg.job.name
        if model_path is None:
            model_path = osp.join(log_dir, "latest_model")
        # Load the saved model
        model = model.load(model_path)

    else:
        # define callbacks to periodically save our model and evaluate it to help monitor training
        # the below freq values will save every 10 rollouts
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=10 * rollout_steps // num_envs,
            deterministic=True,
            render=True,
            n_eval_episodes=10,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=10 * rollout_steps // num_envs,
            save_path=log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # Train an agent with PPO for args.total_timesteps interactions
        model.learn(
            cfg.train.n_steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
        # Save the final model
        model.save(osp.join(log_dir, "latest_model"))

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=True,
        return_episode_rewards=True,
        n_eval_episodes=10,
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
