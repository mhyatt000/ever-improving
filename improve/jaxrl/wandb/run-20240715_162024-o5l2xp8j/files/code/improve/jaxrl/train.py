import os
import os.path as osp
import random
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pprint import pprint
from typing import Any, List, Optional, Tuple

import hydra
import improve
import improve.hydra.resolver
import numpy as np
from improve import cn
from improve.jaxrl.agents import AWACLearner, SACLearner
from improve.jaxrl.datasets import ReplayBuffer
from improve.jaxrl.evaluation import evaluate
from improve.util.config import default, store_as_head
from improve.env import make_envs
from ml_collections import config_flags
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Dataset(Enum):
    D4RL = "d4rl"
    AWAC = "awac"
    RL_UNPLUGGED = "rl_unplugged"


@dataclass
class Job:

    wandb: bool = True  # Track experiments with Weights and Biases.
    wandb_project_name: Optional[str] = 'awac'
    wandb_entity: Optional[str] = None

    log_path: str =  '${r_home:improve_logs}'

    save_video: bool = False  # Save videos during evaluation.
    tqdm: bool = True  # Use tqdm progress bar.

@store_as_head
@dataclass
class MyConfig:

    defaults: List[Any] = default(
        [
            {"algo": "awac"},
            "_self_",
        ]
    )

    env_name: str = "HalfCheetah-v2"  # Environment name.
    dataset_name: Dataset = Dataset.AWAC  # Dataset name.
    seed: int = 42  # Random seed.

    eval_episodes: int = 10  # Number of episodes used for evaluation.
    log_interval: int = 1000  # Logging interval.
    eval_interval: int = 5000  # Eval interval.
    batch_size: int = 256  # Mini batch size.
    updates_per_step: int = 1  # Gradient updates per step.
    max_steps: int = int(1e6)  # Number of training steps.
    start_training: int = int(1e4)  # Number of training steps to start training.

    # Dataset percentile (see https://arxiv.org/abs/2106.01345).
    percentile: float = 100.0
    percentage: float = 100.0  # Percentage of the dataset to use for training.

    job: Job = default(Job())  # Job configuration.
    algo: cn.AWAC = default(cn.AWAC())

    locked: bool = False  # Lock the config file.

    def __post_init__(self):
        self.name = self.algo.name
        print(self.defaults)
        quit()

        # if self.locked:
        # OmegaConf.set_struct(self, True)


def containerize(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)

# @hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
@hydra.main(version_base=None, config_name="config")
def main(cfg: MyConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # 1. set wandb
    # 2. make dataset and video recording
    # 3. set random seed
    # 4. set replay buffer
    # 5. set agent
    # 6. train

    agent_kwargs = containerize(cfg.algo)
    algo = cfg.algo.name
    run_name = f"{cfg.env_name}__{algo}__{cfg.seed}__{int(time.time())}"

    if cfg.job.wandb:
        import wandb

        wandb.init(
            project=cfg.job.wandb_project_name,
            entity=cfg.job.wandb_entity,
            sync_tensorboard=True,
            config=containerize(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        cfg.log_dir = osp.join(cfg.job.log_path, wandb.run.name)

    num_envs = cfg.env.n_envs
    max_episode_steps = cfg.env.max_episode_steps
    rollout_steps = cfg.algo.get("n_steps", None) or 4800

    if cfg.job.seed is not None:
        set_random_seed(cfg.job.seed)

    if cfg.job.wandb.use:
        # initialize wandb logger
        format_strings = ["stdout", "tensorboard"]
        folder = "home/zero-shot/sb3_logs"
        logger = WandbLogger(folder, format_strings)

    # summary_writer = SummaryWriter(os.path.join(cfg.job.log_dir, run_name))

    if cfg.job.save_video:
        video_train_folder = os.path.join(cfg.job.log_dir, "video", "train")
        video_eval_folder = os.path.join(cfg.job.log_dir, "video", "eval")
    else:
        video_train_folder = None
        video_eval_folder = None

    env, eval_env = make_envs(cfg, cfg.job.log_dir, eval_only=False, num_envs=1)
    # env = improve.jaxrl.utils.make_env(cfg.env_name, cfg.seed, video_train_folder)
    # eval_env = improve.jaxrl.utils.make_env(cfg.env_name, cfg.seed + 42, video_eval_folder)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    replay_buffer_size = agent_kwargs.pop("replay_buffer_size")
    obs, act = env.observation_space.sample(), env.action_space.sample()
    obs, act = obs[np.newaxis], act[np.newaxis]

    # 5. set agent
    assert algo.name == "awac"

    if algo.name == "sac":
        agent = SACLearner(cfg.seed, obs, act, **agent_kwargs)
    elif algo.name == "awac":
        agent = AWACLearner(cfg.seed, obs, act, **agent_kwargs)
    else:
        raise NotImplementedError()

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size or cfg.max_steps
    )
    replay_buffer.initialize_with_dataset()

    print("ready to train")
    quit()

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm(range(1, cfg.max_steps + 1), smoothing=0.1, disable=not cfg.job.tqdm):

        if i < cfg.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)

        mask = 1.0 if not done or "TimeLimit.truncated" in info else 0.0

        replay_buffer.insert(
            observation, action, reward, mask, float(done), next_observation
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                logger.record(
                    f"training/{k}", v, info["total"]["timesteps"]
                )

            if "is_success" in info:
                logger.record(
                    f"training/success", info["is_success"], info["total"]["timesteps"]
                )

        if i >= cfg.start_training:
            for _ in range(cfg.updates_per_step):
                batch = replay_buffer.sample(cfg.batch_size)
                update_info = agent.update(batch)

            if i % cfg.log_interval == 0:
                for k, v in update_info.items():
                    logger.record(f"training/{k}", v, i)
                logger.dump(step=i)

        if i % cfg.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, cfg.eval_episodes)

            for k, v in eval_stats.items():
                logger.record(
                    f"evaluation/average_{k}s", v, info["total"]["timesteps"]
                )
            logger.dump(step=i)

            eval_returns.append((info["total"]["timesteps"], eval_stats["return"]))
            np.savetxt(
                os.path.join(cfg.job.log_dir, f"{cfg.seed}.txt"),
                eval_returns,
                fmt=["%d", "%.1f"],
            )


if __name__ == "__main__":
    main()
