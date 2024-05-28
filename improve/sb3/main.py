import warnings
from functools import partial

import gymnasium as gym
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from improve.sb3.mycallback import MyCallback, WandbLogger, ReZeroCallback
from improve.wrappers import residualrl as rrl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

import wandb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def zero_init(model):

    module_gains = {
        model.policy.features_extractor: 0,
        model.policy.mlp_extractor: 0,
        model.policy.action_net: 0,
        model.policy.value_net: 0,
    }
    if not model.policy.share_features_extractor:
        # Note(antonin): this is to keep SB3 results
        # consistent, see GH#1148
        del module_gains[model.policy.features_extractor]
        module_gains[model.policy.pi_features_extractor] = 0
        module_gains[model.policy.vf_features_extractor] = 0

    for module, gain in module_gains.items():
        module.apply(partial(model.policy.init_weights, gain=gain))

    # make the policy.log_std very small
    # model.policy.log_std
    model.policy.log_std.data.fill_(-10)


# TODO cfg should be OmegaConf from hydra
def build_callbacks(env, cfg=None):

    freq = int(1e4)

    checkpoint_callback = CheckpointCallback(
        save_freq=freq,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path="./logs/",
        callback_after_eval=None,
        log_path="./logs/",
        eval_freq=freq,
        deterministic=True,
        render=True,
        verbose=1,
    )

    # wandb_callback = MyCallback()
    rezero = ReZeroCallback(num_reset=50)
    callbacks = CallbackList([checkpoint_callback, eval_callback, rezero])
    return callbacks


def rollout(model):

    images = []
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in tqdm(range(100)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv resets automatically
        if done:
            obs = vec_env.reset()

        # maybe this is why its not learning??
        image = np.transpose(obs["image"].squeeze(), (1, 2, 0))
        images.append(image)

        # TODO no logging dir yet
        # media.write_video(f"{cfg.logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)
        fname = f"./episode_{0}_success_{None}.mp4"
        media.write_video(fname, images, fps=5)

        # wandb log video
        wandb.log({"rollout/video": wandb.Video(fname, fps=5)})


def main():

    env = rrl.make("widowx_put_eggplant_in_basket", kind="sb3")
    # env = rrl.make( "google_robot_pick_horizontal_coke_can", policy="rt1", ckpt="/home/zero-shot/mhyatt000/rtx_rt1x_checkpoint", kind="sb3",)

    # model = A2C("MlpPolicy", env, verbose=1)

    # TODO add cosine schedule
    # not priority
    # torch lr schedule as cosine lr schedule

    policy_kwargs = None
    """
        'lr_schedule':,
        'net_arch': None,
        'activation_fn': nn.Tanh,
        'ortho_init': False,
        'use_sde': False,
        'log_std_init': 0.0,
        'full_std': True,
        'use_expln': False,
        'squash_output': False,
        'features_extractor_class': CombinedExtractor,
        'features_extractor_kwargs': None,
        'share_features_extractor': True,
        'normalize_images': True,
    }
    """

    # TODO put into hydra
    kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 2048,  # was 2048,
        "batch_size": 32,  # was 64 but cuda OOM
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        # clip_range: Union[float, Schedule] = 0.2,
        # clip_range_vf: Union[None, float, Schedule] = None,
        "normalize_advantage": True,
        "ent_coef": 0.00,  # from cleanrl config
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        # rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        # rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        # "target_kl": None,
        "stats_window_size": 100,
        # tensorboard_log: Optional[str] = None,
        # policy_kwargs: Optional[Dict[str, Any]] = None,
        "seed": 0,
        "_init_setup_model": True,
    }

    model = PPO("MultiInputPolicy", env, verbose=1, **kwargs)

    format_strings = ["stdout", "tensorboard"]
    folder = "home/zero-shot/sb3_logs"
    model.set_logger(WandbLogger(folder, format_strings))

    run = wandb.init(
        project="residualrl",
        job_type="train",
        config=kwargs,
    )

    zero_init(model)

    callbacks = build_callbacks(env)

    model.learn(
        total_timesteps=int(1e7),  # was 10_000
        callback=callbacks,
        log_interval=1,
        tb_log_name="PPO",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    rollout(model)

    run.finish()


if __name__ == "__main__":
    main()
