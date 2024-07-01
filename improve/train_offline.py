from datetime import timedelta
from pprint import pprint

import clip
import numpy as np
# import simpler_env as simpler
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs,
                              InitProcessGroupKwargs)
from omegaconf import OmegaConf as OC
from tensordict import TensorDict as TD
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential as TDS
from torch import nn
from torch.nn import functional as F
from torch.profiler import (ProfilerActivity, profile, record_function,
                            tensorboard_trace_handler)
from torch.utils.data import DataLoader, random_split
# from torchrl.collectors import SyncDataCollector
# from torchrl.data.replay_buffers import ReplayBuffer
# from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
# from torchrl.data.replay_buffers.storages import LazyTensorStorage
# from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv)
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import hydra
import improve
import improve.model.vision_transformer as vits
import wandb
from improve.algo import TQC, Algo
from improve.data.flex import HDF5Dataset
from improve.model.hub.gr import GR2
from improve.util.optim import async_step
from improve.util.transform import PreProcess
from improve.wrapper import dict_util as du

# from improve.wrapper.eval import EvalWrapper


class MLP_BNA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU):
        super(MLP_BNA, self).__init__()

        hidden_size = hidden_size

        fc1 = nn.Linear(input_size, hidden_size)
        bn1 = nn.BatchNorm1d(hidden_size)
        act1 = activation()
        fc2 = nn.Linear(hidden_size, output_size)
        bn2 = nn.BatchNorm1d(output_size)
        act2 = activation()

        self.net = nn.ModuleList([fc1, bn1, act1, fc2, bn2, act2])

    def forward(self, x):

        for layer in self.net:
            x = layer(x)
        return x


class MultiInputEncoder(nn.Module):

    def __init__(self, observation):
        super(MultiInputEncoder, self).__init__()
        observation = observation["observation"]

        mlp = lambda x: MLP_BNA(x.shape[-1], 256, 256)
        self.encoder = {
            k: TDM(mlp(v), in_keys=("observation", k), out_keys=("hidden", k))
            for k, v in observation.items()
        }

        self.encoder = TDS(*self.encoder.values())
        self.hidden = nn.Linear(256 * len(observation.keys()), 256)

    def forward(self, x):
        x = self.encoder(x)
        obs = x["hidden"]
        hidden = torch.cat([obs[k] for k in obs.keys()], dim=-1)
        x["latent"] = self.hidden(hidden)
        return x


class ValueHead(nn.Module):

    def __init__(self, hdim, adim, odim):
        super(ValueHead, self).__init__()
        self.net = MLP_BNA(hdim + adim, 256, odim)

    def forward(self, x, act):
        input = torch.cat([x, act], dim=-1)
        return self.net(input)


class DummyActorCritic(nn.Module):

    def __init__(self, observation):
        super(DummyActorCritic, self).__init__()

        self.encoder = MultiInputEncoder(observation)
        self.hidden = 256

        self.critic = TDM(
            ValueHead(self.hidden, 7, 1),
            in_keys=["latent", ("output", "action")],
            out_keys=("output", "value"),
        )
        self.actor = TDM(
            torch.nn.Linear(self.hidden, 7),
            in_keys=["latent"],
            out_keys=("output", "action"),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.actor(x)
        x = self.critic(x)
        return x


class MyCallback:

    def on_train_epoch_end(self, results):
        pass


@hydra.main(config_path=improve.CONFIG, config_name="gr1_config", version_base="1.3.2")
def main(cfg):

    if cfg.job.wandb.use:
        print("Using wandb")
        run = wandb.init(
            project="offline-improvement",
            dir=cfg.callback.log_path,
            job_type="train",
            # sync_tensorboard=True,
            monitor_gym=True,
            name=cfg.job.wandb.name,
            group=cfg.job.wandb.group,
            config=OC.to_container(cfg, resolve=True),
        )
        wandb.config.update({"name": run.name})

    # logger = WandbLogger(root_dir="logs") if cfg.exp.wandb else None

    # The timeout here is 3600s to wait for other processes to finish the simulation
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # amp = FP8RecipeKwargs(backend="msamp", opt_level="O2")
    acc = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        kwargs_handlers=[init_pg_kwargs, ddp_kwargs],
    )
    device = acc.device

    def build_loader(ds):
        return DataLoader(
            ds,
            batch_size=cfg.data.bs_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            pin_memory=True,  
            shuffle=True,
            prefetch_factor=cfg.data.prefetch_factor,
            persistent_workers=True,
            drop_last=True,
        )

    ds = HDF5Dataset(n_steps=1)  # cfg.model.seq_len)
    loader = build_loader(ds)

    batch = next(iter(loader))
    del batch["observation"]["simpler-img"]

    batch = TD(batch, batch_size=cfg.data.bs_per_gpu, device=acc.device)

    """
    model_clip, _ = clip.load(cfg.submodel.clip_backbone)
    model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
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
    """

    model = DummyActorCritic(batch).to(acc.device)

    # TODO read about torch compile (and profiler?)
    # if cfg.training.compile_model:
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_max,
        weight_decay=cfg.training.weight_decay,
        fused=True,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.num_warmup_steps,  # 10_000,  # cfg.num_warmup_epochs * total_prints_per_epoch,
        num_training_steps=cfg.training.num_steps,  # cfg.num_epochs * total_prints_per_epoch,
    )

    toplace = [True, True, True]
    (model, optimizer, loader) = acc.prepare(
        model, optimizer, loader, device_placement=toplace
    )

    algo = TQC(acc, model, loader, optimizer, scheduler, cfg)
    algo.run()

    if cfg.exp.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

batch = {
    "action": (torch.float32, torch.Size([32, 10, 7])),
    "info": {
        "consecutive_grasp": (torch.bool, torch.Size([32, 10])),
        "elapsed_steps": (torch.int64, torch.Size([32, 10])),
        "episode_stats": {
            "consecutive_grasp": (torch.bool, torch.Size([32, 10])),
            "is_src_obj_grasped": (torch.bool, torch.Size([32, 10])),
            "moved_correct_obj": (torch.bool, torch.Size([32, 10])),
            "moved_wrong_obj": (torch.bool, torch.Size([32, 10])),
            "src_on_target": (torch.bool, torch.Size([32, 10])),
        },
        "is_src_obj_grasped": (torch.bool, torch.Size([32, 10])),
        "is_success": (torch.bool, torch.Size([32, 10])),
        "moved_correct_obj": (torch.bool, torch.Size([32, 10])),
        "moved_wrong_obj": (torch.bool, torch.Size([32, 10])),
        "src_on_target": (torch.bool, torch.Size([32, 10])),
        "success": (torch.bool, torch.Size([32, 10])),
        "will_succeed": (torch.bool, torch.Size([32, 10, 1])),
    },
    "observation": {
        "agent_partial-action": (torch.float64, torch.Size([32, 10, 7])),
        "agent_qpos": (torch.float32, torch.Size([32, 10, 8])),
        "agent_qvel": (torch.float32, torch.Size([32, 10, 8])),
        "simpler-img": (torch.uint8, torch.Size([32, 10, 69, 91, 3])),
    },
    "reward": (torch.float64, torch.Size([32, 10])),
    "terminated": (torch.bool, torch.Size([32, 10])),
    "truncated": (torch.bool, torch.Size([32, 10])),
}
