from datetime import timedelta
from pprint import pprint

import clip
import numpy as np

# import simpler_env as simpler
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf as OC
from tensordict import TensorDict as TD
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential as TDS
from torch import nn
from torch.nn import functional as F
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)
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

import lightning as L
from wandb.integration.lightning.fabric import WandbLogger

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MLP_BNA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU):
        super(MLP_BNA, self).__init__()

        hidden_size = hidden_size
        use_bn = True
        bn = nn.LayerNorm if use_bn else nn.Identity

        fc1 = nn.Linear(input_size, hidden_size)
        bn1 = bn(hidden_size)
        act1 = activation()
        fc2 = nn.Linear(hidden_size, output_size)
        bn2 = bn(output_size)
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
        return F.sigmoid(self.net(input))


class DummyActorCritic(nn.Module):

    def __init__(self, observation, name="base"):
        super(DummyActorCritic, self).__init__()

        self.encoder = MultiInputEncoder(observation)
        self.hidden = 256
        self.name = name

        self.critic = TDM(
            ValueHead(self.hidden, 7, 1),
            in_keys=["latent", (name, "action")],
            out_keys=(name, "value"),
        )
        self.actor = TDM(
            torch.nn.Linear(self.hidden, 7),
            in_keys=["latent"],
            out_keys=(name, "action"),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.actor(x)
        x = self.critic(x)
        return x

class MyCallback:

    def on_train_epoch_end(self, results):
        pass

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)




@hydra.main(config_path=improve.CONFIG, config_name="gr1_config", version_base="1.3.2")
def main(cfg):

    torch.set_float32_matmul_precision("medium")

    logger = WandbLogger(
        save_dir=cfg.callback.log_path,
        dir=cfg.callback.log_path,
        project="offline-improvement",
        log_model="all",
        experiment=None,
        prefix="",
        checkpoint_name=None,
        log_checkpoint_on="success",
    )

    print(logger.experiment.config)

    """
    # log gradients and model topology
    wandb_logger.watch(model)

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    # change log frequency of gradients and parameters (100 steps by default)
    wandb_logger.watch(model, log_freq=500)
    """

    fabric = L.Fabric(
        precision="bf16-mixed",
        loggers=[logger],
    )
    fabric.seed_everything(1234)
    fabric.launch()

    if cfg.job.wandb.use:
        print("Using wandb")
        run = wandb.init(
            project="offline-improvement",
            dir=cfg.callback.log_path,
            config=OC.to_container(cfg, resolve=True),
        )
        wandb.config.update({"name": run.name})

    # logger = WandbLogger(root_dir="logs") if cfg.exp.wandb else None

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

    name = "dataset_2024-06-27_165838_google_robot_pick_horizontal_coke_can.h5"
    names = [name]

    ds = HDF5Dataset(names=names, n_steps=10)  # cfg.model.seq_len)
    loader = build_loader(ds)
    loader = fabric.setup_dataloaders(loader)

    batch = next(iter(loader))
    # del batch["observation"]["simpler-img"]

    batch = TD(batch, batch_size=cfg.data.bs_per_gpu)

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

    model = DummyActorCritic(batch, "base").to(fabric.device)
    target = DummyActorCritic(batch, "target").to(fabric.device)

    model.apply(init_weights)
    target.load_state_dict(model.state_dict())

    # TODO read about torch compile (and profiler?)
    # if cfg.training.compile_model:
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_max,
        weight_decay=cfg.training.weight_decay,
        fused=True,
    )
    model, target, optimizer = fabric.setup(model, target, optimizer)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.num_warmup_steps,  # 10_000,  # cfg.num_warmup_epochs * total_prints_per_epoch,
        num_training_steps=cfg.training.num_steps,  # cfg.num_epochs * total_prints_per_epoch,
    )

    model = {
        "base": model,
        "target": target,
    }
    algo = TQC(fabric, model, loader, optimizer, scheduler, cfg)
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
