import os.path as osp
import warnings
from pprint import pprint

import gymnasium as gym
import hydra
import improve
import improve.config.prepare
import improve.config.resolver
import mani_skill2.envs
import numpy as np
import simpler_env as simpler
import stable_baselines3 as sb3
import wandb
from improve.log.wandb import WandbLogger
from improve.sb3 import util
from improve.wrapper import dict_util as du
from improve.wrapper.force_seed import ForceSeedWrapper
from improve.wrapper.normalize import NormalizeObservation, NormalizeReward
from improve.wrapper.sb3.successinfo import SuccessInfoWrapper
from improve.wrapper.simpler import (ExtraObservationWrapper,
                                     FoundationModelWrapper)
from improve.wrapper.simpler.misc import (DownscaleImgWrapper,
                                          FilterKeysWrapper,
                                          FlattenKeysWrapper,
                                          GraspDenseRewardWrapper)
from improve.wrapper.simpler.no_rotation import NoRotationWrapper
from improve.wrapper.simpler.reach_task import ReachTaskWrapper
from improve.wrapper.simpler.rescale import RTXRescaleWrapper
from improve.wrapper.wandb.vec import WandbVecMonitor
from mani_skill2.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf as OC
from stable_baselines3 import A2C, SAC
from improve.sb3.custom import PPO
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor)
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from wandb.integration.sb3 import WandbCallback


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


"""
parser.add_argument(
    "--log-dir",
    type=str,
    default="logs",
    help="path for where logs, checkpoints, and videos are saved",
)
parser.add_argument(
    "--model-path", type=str, help="path to sb3 model for evaluation"
)
"""


def make_env(cfg, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs

        env = simpler.make(
            cfg.env.foundation.task,
            # cant find simpler-img if you specify the mode
            **(
                {"obs_mode": cfg.env.obs_mode.mode}
                if cfg.env.obs_mode.mode != "rgb"
                else {}
            ),
            render_mode="cameras",
            success_from_episode_stats=False,
            max_episode_steps=max_episode_steps,
            renderer_kwargs={
                "offscreen_only": True,
                "device": "cuda:0",
            },
        )

        if cfg.env.foundation.name:
            env = FoundationModelWrapper(
                env,
                task=cfg.env.foundation.task,
                policy=cfg.env.foundation.name,
                ckpt=cfg.env.foundation.ckpt,
                residual_scale=cfg.env.residual_scale,
                action_mask_dims=cfg.env.action_mask_dims,
            )

        env = ExtraObservationWrapper(env)

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

        # must be closer to simpler than rescale
        # this way it overrides the rescale
        if cfg.env.no_quarternion:
            env = NoRotationWrapper(env)

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
            env = RecordEpisode(env, record_dir, info_on_video=True)

        return env

    # if cfg.job.wandb.use:
    # env = WandbInfoStatWrapper(env, logger)

    return _init


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    if cfg.job.wandb.use:
        print("Using wandb")
        run = wandb.init(
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
        wandb.config.update({"name": run.name})

    pprint(OC.to_container(cfg, resolve=True))  # keep after wandb so it logs

    # args = parse_args()
    num_envs = cfg.env.n_envs
    max_episode_steps = cfg.env.max_episode_steps
    log_dir = osp.join(cfg.callback.log_path, wandb.run.name)
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
    if eval_only:
        record_dir = osp.join(log_dir, "videos/eval")
    else:
        record_dir = osp.join(log_dir, "videos")

    if cfg.env.foundation.name is None:
        eval_env = SubprocVecEnv(
            [make_env(cfg, record_dir=record_dir) for _ in range(1)]
        )
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
            if cfg.job.wandb.use:
                env = WandbVecMonitor(env, logger)

            env.seed(cfg.job.seed)
            env.reset()

    if cfg.env.foundation.name:  # using foundation model ... only one env allowed
        print(cfg.env.foundation.name)
        env = DummyVecEnv(
            [
                make_env(
                    cfg,
                    record_dir=record_dir,
                    max_episode_steps=max_episode_steps,
                )
                for _ in range(1)
            ]
        )
        print("made dummy vec env")

        env = VecMonitor(env)
        if cfg.job.wandb.use:
            env = WandbVecMonitor(env, logger)

        print("wrapped env")

        env.seed(cfg.job.seed)
        env.reset()
        eval_env = env

    print(env)

    algo_kwargs = OC.to_container(cfg.algo, resolve=True)
    policy_kwargs = algo_kwargs.get("policy_kwargs", {})
    del algo_kwargs["name"]
    del algo_kwargs["policy_kwargs"]

    policy_kwargs.update(
        {"optimizer_kwargs": {"betas": (0.999, 0.999), "weight_decay": 1e-4}}
    )

    if cfg.algo.name == "sac":
        del algo_kwargs["use_original_space"]
        del algo_kwargs["warmup_zero_action"]

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
        "sac": SAC,
    }[cfg.algo.name]
    model = algo(
        "MultiInputPolicy",
        env,
        **algo_kwargs,
        policy_kwargs=policy_kwargs,
    )

    print(model.policy)

    if cfg.job.wandb.use:
        model.set_logger(logger)

    n_eval = 1 if (cfg.env.seed.force and cfg.env.seed.seeds is None) else 10

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

        eval_callback = EvalCallback(
            eval_env,
            callback_after_eval=None,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=5 * rollout_steps // num_envs,
            deterministic=True,
            render=True,
            n_eval_episodes=n_eval,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10 * rollout_steps // num_envs,
            save_path=log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        callbacks = [checkpoint_callback, eval_callback]

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
