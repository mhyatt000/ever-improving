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
import matplotlib.pyplot as plt
import numpy as np
import simpler_env as simpler
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs,
                              InitProcessGroupKwargs)
from improve.data.flex import HDF5Dataset
from improve.pac.util.optim import async_step
from improve.pac.util.prefetch import DataPrefetcher
from improve.pac.util.transform import PreProcess
from improve.wrapper import dict_util as du
from improve.wrapper.eval import EvalWrapper
from omegaconf import OmegaConf
from torch.profiler import (ProfilerActivity, profile, record_function,
                            tensorboard_trace_handler)
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import models.vision_transformer as vits
import wandb
from models.gr1 import GR1
from models.gr2 import GR2

# Lightning Memory-Mapped Database
# from LMDBDataset_jpeg import LMDBDataset as LMDBdst_jpeg


def plot_values(pred):
    bs, seq, bins = pred.shape
    pred = pred.cpu().numpy()

    # Cumulative probabilities to calculate quantiles.
    quantiles = np.linspace(0, 1, bins) + 0.5 / bins

    plt.switch_backend("Agg")
    # pred = pred.cumsum(-1)
    # print(pred[0,0,-1])
    for b in tqdm(range(bs), desc="plotting values...", leave=False):
        for t in tqdm(range(seq), leave=False):

            plt.title("Quantile Regression over Values")
            plt.ylabel("Estimated Value")
            plt.xlabel("Quantiles - probability space")

            plt.bar(quantiles, pred[b, t], label=f"relative t={t}", width=1 / bins)
            plt.axhline(
                y=pred[b, t].mean(), color="r", linestyle="--", label="mean value"
            )

            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()

            now = datetime.now().strftime("%Y-%m-%d")
            path = osp.join(improve.RESULTS, now)
            os.makedirs(path, exist_ok=True)
            path = osp.join(path, f"value_cdf_sample{b}_t{t}.png")
            plt.savefig(path)
            plt.close()


class Trainer:

    def __init__(self, acc, prefetcher, model, optimizer, scheduler, cfg):

        self.acc = acc
        self.device = self.acc.device
        self.prefetcher = prefetcher
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        self.preprocessor = PreProcess(cn=cfg.data.preprocess, device=self.device)
        self.writer = SummaryWriter(cfg.paths.save_path + "logs")

        self.tokenizer = clip.tokenize
        self.nstep = 0

        if self.cfg.exp.wandb:
            wandb.init(
                project="gr2",
                config=OmegaConf.to_container(cfg),
            )

    def log(self, d, step):
        if not self.cfg.exp.wandb:
            return
        """Log dictionary d to wandb"""
        d = du.apply(d, lambda x: x.cpu().detach() if x is torch.Tensor else x)
        d = du.flatten(d, delim="/")
        wandb.log(d, step=step)

    def save(self):
        self.acc.wait_for_everyone()
        model = self.acc.unwrap_model(self.model)

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.acc.save(
            {"state_dict": state_dict},
            osp.join(improve.WEIGHTS, f"GR1_{self.step}.pth"),
        )

    def load(self):
        path = osp.join(improve.WEIGHTS, self.cfg.paths.ckpt)
        self.model.load_state_dict(torch.load(path)["state_dict"], strict=False)
        self.model = self.acc.prepare(self.model, device_placement=[True])[0]

    def step(self, batch):

        self.model.train()
        self.optimizer.zero_grad()

        """
        'observation': {'agent_partial-action': (torch.float64,
                                              torch.Size([8, 10, 7])),
                     'agent_qpos': (torch.float32, torch.Size([8, 10, 8])),
                     'agent_qvel': (torch.float32, torch.Size([8, 10, 8])),
                     'simpler-img': (torch.uint8,
                                     torch.Size([8, 10, 480, 640, 3]))},
        'reward': (torch.float64, torch.Size([8, 10])),
        """

        # TODO can this be a transform for the dataset?
        # preprocess before prefetching
        img = self.preprocessor._process(
            batch["observation"]["simpler-img"], static=True, train=True
        )

        # xyq quarternions
        # state = batch["observation"]["agent_qpos"]
        # this is wrong
        # state = {"arm": state[:, :7], "gripper": state[:, 7:]}

        # TODO no wrist images rn
        # batch["rgb_static"], batch["rgb_gripper"] = self.preprocessor.rgb_process( batch["rgb_static"], batch["rgb_gripper"], train=True)

        # obs_mask = batch["mask"][..., 0]
        bs, seq = img.shape[:2]
        attn_mask = torch.ones((bs, seq, 1)).to(self.device)

        text = self.tokenizer("put eggplant in the sink").to(self.device)
        text = text.view(1, -1).expand(bs, -1).to(self.device)

        action = batch["observation"]["agent_partial-action"].float()
        obs = {
            "rgb": img,
            # xyz and quarternions for us... or xyz and rpy
            "state": {"arm": action[:, :, :-1], "gripper": action[:, :, -1:]},
            "language": text,
            "mask": attn_mask,
        }

        predictions, targets = self.model(obs)

        action = torch.roll(action, -1, 1).view(bs, seq, 1, -1).repeat(1, 1, 10, 1)
        targets["arm"] = action[..., :-1]

        targets["gripper"] = (action[..., -1:] / 2) + 0.5

        loss = self.model.loss(
            predictions,
            targets,
            batch,
            skip_frame=self.cfg.model_other.skip_frame,
            arm_loss_ratio=self.cfg.training.arm_loss_ratio,
        )

        self.acc.backward(loss["total"])
        self.optimizer.step()
        self.scheduler.step()
        self.log({"loss": loss}, self.nstep)

        if self.nstep % 500 == 0:
            with torch.no_grad():
                plot_values(predictions["value"])

        lr = self.optimizer.param_groups[0]["lr"]
        self.log({"train/lr": lr}, self.nstep)
        self.nstep += 1

        batch, load_time = self.prefetcher.next()
        return batch, load_time

    def rollout(self):
        with torch.no_grad():

            env = EvalWrapper(
                simpler.make(self.cfg.eval.task),
                nstep=self.nstep,
                device=self.device,
                render=True,
            )

            success_rate = []
            lengths = []
            for _ in tqdm(range(10), desc="eval rollouts"):
                obs, info = env.reset()

                process = lambda x: self.preprocessor._process(
                    x.view([1, 1] + list(x.shape)), static=True, train=False
                )
                img = process(obs["rgb"])
                bs, seq = 1, 10

                action = torch.zeros(bs, 1, 7).to(self.device)
                state = {"arm": action[..., :-1], "gripper": action[..., -1:]}
                instruction = env.instruction
                text = self.tokenizer("put eggplant in the sink").to(self.device)
                text = text.view(1, -1).expand(bs, -1).to(self.device)
                attn_mask = torch.ones((bs, seq, 1)).to(self.device)

                buffer = {
                    "rgb": img,
                    "state": state,
                    "mask": torch.ones(1, 1, 1).to(self.device),
                }

                successes = []
                success, truncated, done = False, False, False
                while not done:
                    if seq - buffer["rgb"].shape[1] > 0:
                        n = buffer["rgb"].shape[1]
                        remain = seq - n
                        trajectory = du.apply(
                            buffer,
                            lambda x: torch.zeros_like(x[:, 0]).repeat(
                                [1, remain] + [1 for i in x.shape[2:]]
                            ),
                        )

                        trajectory = du.apply_both(
                            trajectory, buffer, lambda x, y: torch.cat([x, y], dim=1)
                        )

                    else:
                        trajectory = du.apply(buffer, lambda x: x[:, -seq:])

                    # does not repeat over time
                    trajectory["language"] = text

                    predictions, targets = self.model(trajectory)
                    actions = torch.cat(
                        [predictions["arm"], predictions["gripper"]], dim=-1
                    )

                    # this is a hack to undo action chunking until its optimized
                    idx = min(buffer["rgb"].shape[1], seq) - 1
                    action = actions[0, idx, 0, :]

                    obs, reward, success, truncated, info = env.step(action)
                    done = success or truncated
                    successes.append(success)

                    img = process(obs["rgb"])

                    action = action.view(1, 1, -1)
                    timestep = {
                        "rgb": img,
                        "state": {"arm": action[..., :-1], "gripper": action[..., -1:]},
                        "mask": torch.ones(1, 1, 1).to(self.device),
                    }
                    buffer = du.apply_both(
                        buffer, timestep, lambda x, y: torch.cat([x, y], dim=1)
                    )

                lengths.append(len(successes))
                success_rate.append(any(successes))

            success_rate = np.mean(success_rate)
            lengths = np.mean(lengths)

            self.log(
                {"eval": {"mean SR": success_rate, "mean length": lengths}}, self.nstep
            )
            env.close()

    def epoch(self, epoch):

        if False:  # epoch % self.cfg.save_epochs == 0:
            self.save()

        batch, load_time = self.prefetcher.next()
        for _ in tqdm(range(len(self.prefetcher.loader)), desc="epoch"):
            with self.acc.accumulate(self.model):
                batch, load_time = self.step(batch)
        self.rollout()

    def run(self):

        # by epochs
        if self.cfg.training.num_epochs > 0:
            for epoch in range(self.cfg.training.num_epochs):
                self.epoch(epoch)
                # self.save()
                self.log({"time/epoch": epoch}, self.nstep)
                # self.Logger.log()

        # by steps
        else:
            epoch = 0
            for _ in tqdm(
                range(self.cfg.training.num_steps // len(self.prefetcher.loader)),
                desc="steps",
            ):
                self.epoch(epoch)
                self.log({"time/epoch": epoch}, self.nstep)
                epoch += 1
                # self.Logger.log()


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


@hydra.main(
    config_path=osp.dirname(__file__), config_name="gr1_config", version_base="1.3.2"
)
def main(cfg):

    # The timeout here is 3600s to wait for other processes to finish the simulation
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        kwargs_handlers=[init_pg_kwargs, ddp_kwargs],
    )
    device = acc.device

    ds = HDF5Dataset(n_steps=cfg.model.seq_len)

    # change shuffle to True somehow
    def build_loader(ds):
        return DataLoader(
            ds,
            batch_size=cfg.data.bs_per_gpu,  # to be flattened in prefetcher
            num_workers=cfg.data.workers_per_gpu,
            pin_memory=True,  # Accelerate data reading
            shuffle=True,
            prefetch_factor=cfg.data.prefetch_factor,
            persistent_workers=True,
        )

    loader = build_loader(ds)

    # for batch in loader:
    # print("batch", du.apply(batch, lambda x: (x.dtype, x.shape)))

    # test_loader = build_loader(test_dataset)

    model_clip, _ = clip.load(cfg.submodel.clip_backbone, device=device)
    model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0).to(device)
    checkpoint = torch.load(osp.join(improve.WEIGHTS, cfg.paths.mae_ckpt))
    model_mae.load_state_dict(checkpoint["model"], strict=False)

    # to device for fused optimizer
    model = GR2.from_hydra(
        cn=cfg.model,
        pretrained={
            "language": model_clip,
            "visual": model_mae,
        },
    ).to(device)

    # maybe_load(model, acc, cfg)
    step = 0  # maybe_resume(model, acc, cfg)

    if cfg.training.compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_max,
        weight_decay=cfg.training.weight_decay,
        fused=True,
    )

    # total_prints_per_epoch = len(train_dataset) // ( cfg.print_steps * cfg.bs_per_gpu * acc.num_processes)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,  # 10_000,  # cfg.num_warmup_epochs * total_prints_per_epoch,
        num_training_steps=cfg.training.num_steps,  # cfg.num_epochs * total_prints_per_epoch,
    )

    (model, optimizer, loader) = acc.prepare(
        model,
        optimizer,
        loader,
        device_placement=[True, True, False],
    )

    # optimizer.step = async_step
    prefetcher = DataPrefetcher(loader, device)
    # test_prefetcher = DataPrefetcher(test_loader, device)

    T = Trainer(acc, prefetcher, model, optimizer, scheduler, cfg)
    T.run()

    if cfg.exp.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
