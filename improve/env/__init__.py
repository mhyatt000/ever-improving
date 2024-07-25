import os.path as osp

import gymnasium as gym
import simpler_env as simpler
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor, VecVideoRecorder)

import improve.wrapper as W  # TODO add all the wrappers to wrapper.__init__.py

from .action_rescale import ActionRescaler

MULTI_OBJ_ENVS = [
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    # "widowx_put_eggplant_in_basket",
]


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


def make_env(cfg, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        # import mani_skill2.envs

        extra = {}

        ### FIX THIS
        if cfg.env.obs_mode.mode.value != "rgb":
            extra["obs_mode"] = cfg.env.obs_mode.mode.value
        if cfg.env.task == "google_robot_pick_horizontal_coke_can":
            extra["success_from_episode_stats"] = False

        env = simpler.make(
            cfg.env.foundation.task,
            # cant find simpler-img if you specify the mode
            render_mode="cameras",
            max_episode_steps=max_episode_steps,
            renderer_kwargs={
                "offscreen_only": True,
                "device": "cuda:0",
            },
            **extra,
        )

        if cfg.algo.name == "awac" or cfg.env.foundation.name is None:
            env = W.ActionRescaleWrapper(env)
            env = W.AwacRewardWrapper(env)
            print("shifting reward dist to [-1, 0]")

        if cfg.env.fm_loc.value == "env":
            if cfg.env.foundation.name:
                env = W.FoundationModelWrapper(
                    env,
                    task=cfg.env.foundation.task,
                    policy=cfg.env.foundation.name,
                    ckpt=cfg.env.foundation.ckpt,
                    residual_scale=cfg.env.residual_scale,
                    strategy=cfg.env.scale_strategy,
                )

            if cfg.env.action_mask_dims:
                env = W.ActionSpaceWrapper(env, cfg.env.action_mask_dims)

        env = W.StickyGripperWrapper(env, task=cfg.env.foundation.task)
        env = W.ExtraObservationWrapper(
            env,
            use_image=cfg.env.obs_mode.mode.value == "rgb",
        )

        if cfg.env.foundation.task in MULTI_OBJ_ENVS:
            print("using src tgt wrapper")
            env = W.SourceTargetWrapper(env)

        if "drawer" in cfg.env.foundation.task:
            print("using drawer wrapper")
            env = W.DrawerWrapper(env)

        if cfg.env.seed.force:
            if cfg.env.seed.seeds is not None:
                env = W.ForceSeedWrapper(env, seeds=cfg.env.seed.seeds, verbose=True)
            else:
                env = W.ForceSeedWrapper(env, seed=cfg.env.seed.value, verbose=True)

        env = W.FlattenKeysWrapper(env)
        if cfg.env.obs_keys:
            env = W.FilterKeysWrapper(env, keys=cfg.env.obs_keys)

        # dont need this wrapper if not using grasp task
        if cfg.env.reward == "dense" and not cfg.env.reach:
            env = W.GraspDenseRewardWrapper(env, clip=0.2)

        if cfg.env.downscale != 1:
            env = W.DownscaleImgWrapper(env, downscale=cfg.env.downscale)

        # NOTE: replaced by W.ActionSpaceWrapper since it is more general
        # must be closer to simpler than rescale
        # this way it overrides the rescale
        # if cfg.env.no_quarternion:
        # env = W.NoRotationWrapper(env)

        if cfg.env.fm_loc.value == "env":  # otherwise rescale is done in the algo
            if cfg.env.scale_strategy == "clip":
                env = W.RTXRescaleWrapper(env)

        if cfg.env.reach:
            env = W.ReachTaskWrapper(
                env,
                use_sparse_reward=cfg.env.reward == "sparse",
                thresh=0.05,
                reward_clip=0.2,
            )

        env = W.SuccessInfoWrapper(env)

        # env = W.WandbActionStatWrapper( env, logger, names=["x", "y", "z", "rx", "ry", "rz", "gripper"],)

        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the W.ContinuousTaskWrapper here for that
        # if max_episode_steps is not None:
        # env = W.ContinuousTaskWrapper(env)

        # if record_dir is not None:
        # print(f"TODO RECORD: {record_dir}")
        # env = RecordEpisode(env, record_dir, info_on_video=True)

        # print(env.observation_space)

        return env

    # if cfg.job.wandb.use:
    # env = WandbInfoStatWrapper(env, logger)

    return _init


def make_envs(cfg, log_dir, eval_only=False, num_envs=1, max_episode_steps=60):

    suffix = "eval" if eval_only else "train"
    record_dir = osp.join(log_dir, f"videos/{suffix}") if cfg.job.wandb.use else None

    if cfg.env.foundation.name is None or cfg.env.fm_loc.value == "central":
        venv = SubprocVecEnv(
            [make_env(cfg, record_dir=record_dir) for _ in range(num_envs)]
        )
        venv = VecMonitor(venv)  # attach this so SB3 can log reward metrics

        venv.seed(cfg.job.seed)
        venv.reset()

        if not cfg.job.wandb.use:  # if not using wandb, dont record anything
            env, eval_env = venv, venv
            return env, eval_env

        eval_env = W.VecRecord(venv, osp.join(log_dir, "eval"), use_wandb=True)
        if not cfg.env.record:  # if not recording for offline, only wrap eval
            return venv, eval_env

        env = W.VecRecord(venv, osp.join(log_dir, "train"), use_wandb=True)
        return env, eval_env

        # this was from maniskill2
        # return eval_env, VecVideoRecorder(eval_env, record_dir, record_video_trigger=lambda x: True)

    raise NotImplementedError("No eval only until this function is fixed.")

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
        # if cfg.job.wandb.use:
        # env = W.WandbVecMonitor(env, logger)

        env.seed(cfg.job.seed)
        env.reset()
    return env, eval_env

    raise NotImplementedError("Only foundation model allowed for now.")
    """
    if cfg.env.foundation.name and cfg.env.fm_loc.value == "env":
        num = 1 if cfg.env.fm_loc.value == "env" else num_envs
        print(cfg.env.foundation.name)
        env = SubprocVecEnv(
            [
                make_env(
                    cfg,
                    record_dir=record_dir,
                    max_episode_steps=max_episode_steps,
                )
                for _ in range(num)
            ]
        )
        print("made dummy vec env")

        env = VecMonitor(env)
        # if cfg.job.wandb.use:
        # env = W.WandbVecMonitor(env, logger)
        
        # add dataset recorder
        if cfg.env.record:
            env = W.VecRecord(env, log_dir, use_wandb=True)
            print("recording data")

        print("wrapped env")

        env.seed(cfg.job.seed)
        env.reset()
        eval_env = env

        return env, eval_env
    """
