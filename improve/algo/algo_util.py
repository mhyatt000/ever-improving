import json
import math
import os
import os.path as osp
from datetime import datetime, timedelta
from pprint import pprint
from time import time

import clip
import hydra
import improve
import improve.model.vision_transformer as vits
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import simpler_env as simpler
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs,
                              InitProcessGroupKwargs)
from improve.data.flex import HDF5Dataset
from improve.model.hub.gr import GR2
from improve.util.optim import async_step
from improve.util.prefetch import DataPrefetcher
from improve.util.transform import PreProcess
from improve.wrapper import dict_util as du
from improve.wrapper.eval import EvalWrapper
from omegaconf import OmegaConf
from torch.profiler import (ProfilerActivity, profile, record_function,
                            tensorboard_trace_handler)
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup





def plot_values(pred, nstep):
    bs, seq, bins = pred.shape
    pred = pred.cpu().numpy()

    # Cumulative probabilities to calculate quantiles.
    quantiles = np.linspace(0, 1, bins) + 0.5 / bins

    plt.switch_backend("Agg")
    # pred = pred.cumsum(-1)
    # print(pred[0,0,-1])
    for b in tqdm(range(bs), desc="plotting values...", leave=False):
        for t in tqdm(range(seq), leave=False):

            fig, axs = plt.subplots(2, 1, figsize=(10, 5))
            fig.suptitle("Quantile Regression over Values")
            axs[0].set_ylabel("Estimated Value")
            axs[0].set_xlabel("Quantiles - probability space")

            axs[0].bar(quantiles, pred[b, t], label=f"relative t={t}", width=1 / bins)
            axs[0].axhline(
                y=pred[b, t].mean(), color="r", linestyle="--", label="mean value"
            )

            axs[0].set_ylim(0, 1)
            axs[0].legend()

            axs[1].hist(pred[b, t], bins=bins, alpha=0.75, color='g')
            axs[1].set_xlim(0, 1)
            axs[1].set_ylim(0, 1)
            axs[1].set_ylabel("Likelihood")
            axs[1].set_xlabel("Estimated Value")

            plt.tight_layout()

            now = datetime.now().strftime("%Y-%m-%d")
            dirname = osp.join(improve.RESULTS, now)
            os.makedirs(dirname, exist_ok=True)
            path = osp.join(dirname, f"value_cdf_sample{b}_t{t}.png")
            plt.savefig(path)
            plt.close()

    # for all the batches, load the figures as np array and save them as a gif
    videos = []
    for b in range(bs):
        paths = [osp.join(dirname, f"value_cdf_sample{b}_t{t}.png") for t in range(seq)]
        path = osp.join(dirname, f"value_cdf_sample{b}.gif")
        imgs = [plt.imread(p)[..., :3] for p in paths]
        mediapy.write_video(path, imgs, fps=5, codec="gif")
        videos.append(path)

        wandb.log({f"cdf/value_samples": [wandb.Video(p) for p in videos]}, step=nstep)




def print_model_params(model, trainable=True):
    if trainable:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of trainable={trainable} parameters: {total_params:_d}")


def maybe_load(model, acc, cfg):
    if cfg.paths.ckpt:
        path = osp.join(improve.WEIGHTS, cfg.paths.ckpt)
        model.load_state_dict(torch.load(path)["state_dict"], strict=False)
        acc.print("load ", path)


def maybe_resume(model, acc, cfg):
    if os.path.isfile(cfg.paths.save_path + "GR1_{}.pth".format(cfg.load_epoch)):
        state_dict = torch.load(
            cfg.paths.save_path + "GR1_{}.pth".format(cfg.load_epoch)
        )["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        acc.print(f"load {cfg.paths.save_path} GR1_{cfg.load_epoch}.pth")

    if os.path.isfile(cfg.paths.save_path + "step.json"):
        with open(cfg.paths.save_path + "step.json", "r") as json_file:
            step = json.load(open(cfg.paths.save_path + "step.json"))
    else:
        step = 0

    return step



