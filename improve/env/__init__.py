import os.path as osp

import gymnasium as gym
import simpler_env as simpler
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor, VecVideoRecorder)

from improve.wrapper.force_seed import ForceSeedWrapper
from improve.wrapper.normalize import NormalizeObservation, NormalizeReward
from improve.wrapper.sb3.successinfo import SuccessInfoWrapper
from improve.wrapper.simpler import (ActionSpaceWrapper,
                                     ExtraObservationWrapper,
                                     FoundationModelWrapper)
from improve.wrapper.simpler.misc import (DownscaleImgWrapper,
                                          FilterKeysWrapper,
                                          FlattenKeysWrapper,
                                          GraspDenseRewardWrapper)
from improve.wrapper.simpler.no_rotation import NoRotationWrapper
from improve.wrapper.simpler.reach_task import ReachTaskWrapper
from improve.wrapper.simpler.rescale import RTXRescaleWrapper
from improve.wrapper.simpler.source_target import SourceTargetWrapper
from improve.wrapper.wandb.record import VecRecord
from improve.wrapper.wandb.vec import WandbVecMonitor

from .action_rescale import ActionRescaler

MULTI_OBJ_ENVS = [
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
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

        if cfg.env.obs_mode.mode != "rgb":
            extra["obs_mode"] = cfg.env.obs_mode.mode
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

        if cfg.env.fm_loc == "env":

            if cfg.env.foundation.name:
                env = FoundationModelWrapper(
                    env,
                    task=cfg.env.foundation.task,
                    policy=cfg.env.foundation.name,
                    ckpt=cfg.env.foundation.ckpt,
                    residual_scale=cfg.env.residual_scale,
                    strategy=cfg.env.scale_strategy,
                )

            if cfg.env.action_mask_dims:
                env = ActionSpaceWrapper(env, cfg.env.action_mask_dims)

        env = ExtraObservationWrapper(env)

        if cfg.env.foundation.task in MULTI_OBJ_ENVS:
            env = SourceTargetWrapper(env)

        if cfg.env.seed.force:
            if cfg.env.seed.seeds is not None:
                env = ForceSeedWrapper(env, seeds=cfg.env.seed.seeds, verbose=True)
            else:
                env = ForceSeedWrapper(env, seed=cfg.env.seed.value, verbose=True)

        env = FlattenKeysWrapper(env)
        if cfg.env.obs_keys:
            env = FilterKeysWrapper(env, keys=cfg.env.obs_keys)

        # dont need this wrapper if not using grasp task
        if cfg.env.reward == "dense" and not cfg.env.reach:
            env = GraspDenseRewardWrapper(env, clip=0.2)

        if cfg.env.downscale != 1:
            env = DownscaleImgWrapper(env, downscale=cfg.env.downscale)

        # NOTE: replaced by ActionSpaceWrapper since it is more general
        # must be closer to simpler than rescale
        # this way it overrides the rescale
        # if cfg.env.no_quarternion:
        # env = NoRotationWrapper(env)

        if cfg.env.fm_loc == "env":  # otherwise rescale is done in the algo
            if cfg.env.scale_strategy == "clip":
                env = RTXRescaleWrapper(env)

        if cfg.env.reach:
            env = ReachTaskWrapper(
                env,
                use_sparse_reward=cfg.env.reward == "sparse",
                thresh=0.05,
                reward_clip=0.2,
            )

        env = SuccessInfoWrapper(env)

        # env = WandbActionStatWrapper( env, logger, names=["x", "y", "z", "rx", "ry", "rz", "gripper"],)

        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        # if max_episode_steps is not None:
        # env = ContinuousTaskWrapper(env)

        if record_dir is not None:
            print(f"TODO RECORD: {record_dir}")
            # env = RecordEpisode(env, record_dir, info_on_video=True)

        # print(env.observation_space)

        return env

    # if cfg.job.wandb.use:
    # env = WandbInfoStatWrapper(env, logger)

    return _init


def make_envs(cfg, log_dir, eval_only=False, num_envs=1, max_episode_steps=None):

    suffix = "eval" if eval_only else "train"
    record_dir = osp.join(log_dir, f"videos/{suffix}")

    if cfg.env.foundation.name is None or cfg.env.fm_loc == "central":
        eval_env = SubprocVecEnv(
            [make_env(cfg, record_dir=record_dir) for _ in range(num_envs)]
        )
        eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
        eval_env.seed(cfg.job.seed)
        eval_env.reset()

        return (
            # big slow down
            # VecRecord( eval_env, osp.join(log_dir, "train"), use_wandb=False,),
            eval_env,
            VecRecord(
                eval_env,
                osp.join(log_dir, "eval"),
                use_wandb=True,
            ),
        )

        # return eval_env, VecVideoRecorder(eval_env, record_dir, record_video_trigger=lambda x: True)

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
            # env = WandbVecMonitor(env, logger)

            env.seed(cfg.job.seed)
            env.reset()
        return env, eval_env

    # using foundation model ... only one env allowed
    if cfg.env.foundation.name and cfg.env.fm_loc == "env":
        num = 1 if cfg.env.fm_loc == "env" else num_envs
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
        # env = WandbVecMonitor(env, logger)

        print("wrapped env")

        env.seed(cfg.job.seed)
        env.reset()
        eval_env = env

        return env, eval_env
