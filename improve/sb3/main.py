import os
from improve.wrapper.simpler.reach_task import ReachTaskWrapper
from improve.wrapper.simpler.rescale import RTXRescaleWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import math
import os.path as osp
import warnings
from functools import partial
from pprint import pprint

import gymnasium as gym
from improve.wrapper.probe import ProbeEnv
import hydra
import improve
import improve.hydra.resolver
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from improve.sb3 import util
from improve.sb3.custom.sac import SAC
from improve.sb3.util import MyCallback, ReZeroCallback, WandbLogger
from improve.wrapper import residualrl as rrl
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from stable_baselines3 import A2C, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

from improve.wrapper.normalize import NormalizeObservation, NormalizeReward

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# TODO cfg should be OmegaConf from hydra
def build_callbacks(env, cfg):

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.callback.freq,
        save_path=cfg.callback.log_path,
        name_prefix=cfg.callback.ckpt.name_prefix,
        save_replay_buffer=cfg.callback.ckpt.save_others,
        save_vecnormalize=cfg.callback.ckpt.save_others,
    )

    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=cfg.callback.log_path,
        callback_after_eval=None,
        log_path=cfg.callback.log_path,
        eval_freq=cfg.callback.freq,
        deterministic=True,
        render=True,  # TODO does this work?
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback]

    if cfg.job.wandb.use:
        wandbCb = WandbCallback(
            gradient_save_freq=1,
            log="gradients",
            verbose=2,
        )
        callbacks += [wandbCb]

    if cfg.callback.rezero.use:
        rezero = ReZeroCallback(
            cfg.algo.name,
            num_reset=cfg.callback.rezero.num_reset,
        )
        callbacks.append(rezero)

    callbacks = CallbackList(callbacks)
    return callbacks


def build_algo(_cfg):
    algos = {
        "ppo": PPO,
        "sac": SAC,
        "a2c": A2C,
    }
    return algos[_cfg.name]


def rollout(model):

    for ep_id in range(1):

        images = []
        vec_env = model.get_env()
        obs = vec_env.reset()

        for i in tqdm(range(130)):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            # maybe this is why its not learning??
            image = np.transpose(obs["image"].squeeze(), (1, 2, 0))
            images.append(image)

            # VecEnv resets automatically
            if done:
                break

        # TODO no logging dir yet
        # media.write_video(f"{cfg.logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)
        fname = f"./episode_{0}_success_{None}.mp4"
        media.write_video(fname, images, fps=5)

        # wandb log video
        wandb.log({f"video/{ep_id}": wandb.Video(fname, fps=5)})



class SuccessInfoWrapper(gym.Wrapper):
    """ A simple wrapper that adds a is_success key which SB3 tracks"""
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    if cfg.job.wandb.use:
        wandb.init(
            project="residualrl",
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

    def make_env(cfg, max_episode_steps: int = None, record_dir: str = None):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            # this was for maniskill2 only
            import improve.wrapper.residualrl as rrl

            env = rrl.make(
                cfg.env,
                obs_mode="state_dict",
                render_mode="cameras",
                max_episode_steps=max_episode_steps,
                renderer_kwargs={
                    "offscreen_only": True,
                    "device": "cuda:0",
                },
            )

            env = RTXRescaleWrapper(env)
            if cfg.env.reach:
                env = ReachTaskWrapper(env, use_sparse_reward=False, thresh=0.01)

            # env = WandbActionStatWrapper( env, logger, names=["x", "y", "z", "rx", "ry", "rz", "gripper"],)

            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that

            # if max_episode_steps is not None:
            # env = ContinuousTaskWrapper(env)
            env = SuccessInfoWrapper(env)
            if record_dir is not None:
                env = RecordEpisode(env, record_dir, info_on_video=True)
            return env

        env = NormalizeObservation(NormalizeReward(env))
        # if cfg.job.wandb.use:
        # env = WandbInfoStatWrapper(env, logger)

        return _init

    eval_only = not cfg.train.use_train
    # create eval environment
    record_dir = osp.join(log_dir, f"videos{'/eval' if eval_only else ''}")

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

    # if cfg.env.goal.use:  env = cfg.env.goal.cls(env, cfg.env.goal.key)

    learning_rate = cfg.algo.learning_rate
    del cfg.algo.learning_rate
    betas = (0.999, 0.999)

    algo = build_algo(cfg.algo)
    algo_kwargs = OC.to_container(cfg.algo, resolve=True)
    policy_kwargs = algo_kwargs.get("policy_kwargs", {})
    del algo_kwargs["name"]
    del algo_kwargs["policy_kwargs"]

    model = algo(
        "MultiInputPolicy",
        env,
        verbose=1,
        **algo_kwargs,
        policy_kwargs={
            "optimizer_kwargs": {
                "betas": betas,
                "weight_decay": 1e-4,
                # "lr_schedule": coslr, ... doesnt seem to accept
            },
            **policy_kwargs,
        },
        learning_rate=learning_rate,
    )

    if cfg.train.use_zero_init:
        util.zero_init(model, cfg.algo.name)

    if cfg.job.wandb.use:
        # initialize wandb logger
        format_strings = ["stdout", "tensorboard"]
        folder = "home/zero-shot/sb3_logs"
        model.set_logger(WandbLogger(folder, format_strings))

    callbacks = build_callbacks(env, cfg)

    if cfg.train.use_train:
        model.learn(
            total_timesteps=cfg.train.n_steps,
            callback=callbacks,
            log_interval=1,
            tb_log_name=cfg.algo.name,
            reset_num_timesteps=True,
            progress_bar=True,
        )

    # rollout(model)
    env.close()

    if cfg.job.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    main()
