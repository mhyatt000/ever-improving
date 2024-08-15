import os
from dataclasses import dataclass
from enum import Enum

import hydra
import numpy as np
import tqdm
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import BCLearner
from jaxrl.datasets import make_env_and_dataset
from jaxrl.evaluation import evaluate


class Dataset(Enum):
    D4RL = "d4rl"
    AWAC = "awac"
    RL_UNPLUGGED = "rl_unplugged"


@dataclass
class Config:
    env_name: str = "halfcheetah-expert-v2"  # Environment name.
    dataset_name: Dataset = Dataset.D4RL  # Dataset name.
    save_dir: str = "./tmp/"  # Tensorboard logging dir.
    seed: int = 42  # Random seed.
    eval_episodes: int = 10  # Number of episodes used for evaluation.
    log_interval: int = 1000  # Logging interval.
    eval_interval: int = 5000  # Eval interval.
    batch_size: int = 256  # Mini batch size.
    max_steps: int = int(1e6)  # Number of training steps.
    # Dataset percentile (see https://arxiv.org/abs/2106.01345).
    percentile: float = 100.0
    percentage: float = 100.0  # Percentage of the dataset to use for training.
    tqdm: bool = True  # Use tqdm progress bar.
    save_video: bool = False  # Save videos during evaluation.
    # File path to the training hyperparameter configuration.
    config: str = "configs/bc_default.py"


@hydra.main(config_path="configs", config_name="config")
def main(cfg: Config):

    summary_writer = SummaryWriter(os.path.join(cfg.save_dir, "tb", str(cfg.seed)))

    video_save_folder = (
        None if not cfg.save_video else os.path.join(cfg.save_dir, "video", "eval")
    )

    env, dataset = make_env_and_dataset(
        cfg.env_name, cfg.seed, cfg.dataset_name, video_save_folder
    )

    if cfg.percentage < 100.0:
        dataset.take_random(cfg.percentage)

    if cfg.percentile < 100.0:
        dataset.take_top(cfg.percentile)

    kwargs = dict(cfg.config)
    kwargs["num_steps"] = cfg.max_steps
    agent = BCLearner(
        cfg.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        **kwargs,
    )

    eval_returns = []
    for i in tqdm.tqdm(
        range(1, cfg.max_steps + 1), smoothing=0.1, disable=not cfg.tqdm
    ):
        batch = dataset.sample(cfg.batch_size)

        update_info = agent.update(batch)

        if i % cfg.log_interval == 0:
            for k, v in update_info.items():
                summary_writer.add_scalar(f"training/{k}", v, i)
            summary_writer.flush()

        if i % cfg.eval_interval == 0:
            eval_stats = evaluate(agent, env, cfg.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{k}s", v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats["return"]))
            np.savetxt(
                os.path.join(cfg.save_dir, f"{cfg.seed}.txt"),
                eval_returns,
                fmt=["%d", "%.1f"],
            )


if __name__ == "__main__":
    main()
