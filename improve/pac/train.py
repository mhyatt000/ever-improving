import json
import math
import os
import os.path as osp
from datetime import timedelta
from pprint import pprint
from time import time

import clip
import hydra
import improve
import numpy as np
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


def save_model(acc, model, cfg, epoch, modules_to_exclude=["model_mae", "model_clip"]):
    acc.wait_for_everyone()
    unwrapped_model = acc.unwrap_model(model)

    if hasattr(unwrapped_model, "_orig_mod"):
        state_dict = {
            k: v
            for k, v in unwrapped_model._orig_mod.state_dict().items()
            if not any(module_name in k for module_name in modules_to_exclude)
        }
    else:
        state_dict = {
            k: v
            for k, v in unwrapped_model.state_dict().items()
            if not any(module_name in k for module_name in modules_to_exclude)
        }

    acc.save(
        {"state_dict": state_dict},
        cfg.save_path + "GR1_{}.pth".format(epoch + cfg.load_epoch),
    )


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

        wandb.init(
            project="gr2",
            config=OmegaConf.to_container(cfg),
        )

    def log(self, d, step):
        """Log dictionary d to wandb"""
        d = du.apply(d, lambda x: x.cpu().detach() if x is torch.Tensor else x)
        d = du.flatten(d, delim="/")
        wandb.log(d, step=step)

    def save(self):
        save_model(
            self.acc,
            self.model,
            self.cfg,
            self.epoch,
            modules_to_exclude=["model_mae", "model_clip"],
        )

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

        _batch = batch

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
        batch = {
            "rgb": img,
            # xyz and quarternions for us... or xyz and rpy
            "state": {"arm": action[:, :, :-1], "gripper": action[:, :, -1:]},
            "language": text,
            "mask": attn_mask,
        }

        predictions, targets = self.model(batch)

        action = torch.roll(action, -1, 1).view(bs, seq, 1, -1).repeat(1, 1, 10, 1)
        targets["arm"] = action[..., :-1]

        targets["gripper"] = (action[..., -1:] / 2) + 0.5

        loss = self.model.loss(
            predictions,
            targets,
            _batch,
            skip_frame=self.cfg.model_other.skip_frame,
            arm_loss_ratio=self.cfg.training.arm_loss_ratio,
        )

        self.acc.backward(loss["total"])
        self.optimizer.step()
        self.scheduler.step()
        self.log({"loss": loss}, self.nstep)
        self.nstep += 1

        lr = self.optimizer.param_groups[0]["lr"]
        self.log({"train/lr": lr}, self.nstep)

        batch, load_time = self.prefetcher.next()
        return batch, load_time

    def epoch(self, epoch):

        if False:  # epoch % self.cfg.save_epochs == 0:
            self.save()

        batch, load_time = self.prefetcher.next()
        while batch is not None:
            with self.acc.accumulate(self.model):
                batch, load_time = self.step(batch)

    def run(self):

        # by epochs
        if self.cfg.training.num_epochs > 0:
            for epoch in range(self.cfg.training.num_epochs):
                self.epoch(epoch)
                self.log({"time/epoch": epoch}, self.nstep)
                # self.Logger.log()

        # by steps
        else:
            epoch = 0
            while self.nstep < self.cfg.training.num_steps:
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
        num_training_steps=6_000,  # cfg.num_epochs * total_prints_per_epoch,
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

    wandb.finish()


if __name__ == "__main__":
    main()
