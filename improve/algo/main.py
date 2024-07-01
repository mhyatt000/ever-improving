
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




@hydra.main(config_path=improve.CONFIG, config_name="gr1_config", version_base="1.3.2")
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
        num_warmup_steps=cfg.training.num_warmup_steps,  # 10_000,  # cfg.num_warmup_epochs * total_prints_per_epoch,
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

    A = Algo(acc, prefetcher, model, optimizer, scheduler, cfg)
    A.run()

    if cfg.exp.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
