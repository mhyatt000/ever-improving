import os
from improve.util.timer import Timer, timer
import os.path as osp

import clip
import numpy as np

# import simpler_env as simpler
import torch
from omegaconf import OmegaConf as OC
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hydra
import improve
import wandb
from improve.util.transform import PreProcess
from improve.wrapper import dict_util as du


class Algo:
    """Base Training Algorithm"""

    def __init__(self, fabric, model, loader, optimizer, scheduler, cfg):

        self.fabric = fabric
        self.device = self.fabric.device
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.dir = cfg.callback.log_path

        self.preprocessor = PreProcess(cn=cfg.data.preprocess, device=self.device)
        self.writer = SummaryWriter(cfg.paths.save_path + "logs")

        self.tokenizer = clip.tokenize
        self.nstep = 0

        self.state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "nstep": self.nstep,
        }

    def log(self, d, step):
        """Log dictionary d to wandb"""
        d = du.apply(d, lambda x: x.cpu().detach() if x is torch.Tensor else x)
        d = du.flatten(d, delim="/")
        self.fabric.log_dict(d, step=step)


    def save(self):
        fname = osp.join(self.dir, "checkpoint.ckpt")
        self.fabric.save(fname, self.state)

    def load(self):
        raise NotImplementedError
        path = osp.join(improve.WEIGHTS, self.cfg.paths.ckpt)
        self.model.load_state_dict(torch.load(path)["state_dict"], strict=False)
        self.model = self.acc.prepare(self.model, device_placement=[True])[0]

    def loss(self, x):
        raise NotImplementedError

    def transform_batch(self, batch):
        return batch

    def maybe_backward(self, loss):
        if True:
            self.fabric.backward(loss)
            self.fabric.clip_gradients(self.model, self.optimizer, max_norm=2.0)

            self.optimizer.step()
            self.scheduler.step()

        else:  # accumulate gradient ... not ready yet

            raise NotImplementedError
            # Accumulate gradient 8 batches at a time
            acc_every = self.cfg.training.gradient_accumulation_steps
            acc = self.nstep % acc_every != 0

            with self.fabric.no_backward_sync(self.model, enabled=acc):
                output = self.model(input)
                loss = loss

                self.fabric.backward(loss)

            if not acc:
                # Step the optimizer after accumulation phase is over
                self.optimizer.step()
                self.optimizer.zero_grad()

    def step(self, batch):

        self.model.train()
        self.optimizer.zero_grad()

        batch = self.transform_batch(batch)
        batch = self.model(batch)
        loss = self.loss(batch)
        self.maybe_backward(loss["total"])

        # housekeeping
        self.log({"loss": loss}, self.nstep)

        lr = self.optimizer.param_groups[0]["lr"]
        self.log({"train/lr": lr}, self.nstep)
        self.nstep += 1

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
            for _ in tqdm(range(10), desc="eval rollouts", leave=False):
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
                    # they are normal distributions now
                    actions = torch.cat(
                        [predictions["arm"].mean, predictions["gripper"].mean], dim=-1
                    )

                    # this is a hack to undo action chunking until its optimized
                    # TODO let model select best of n proposals
                    idx = min(buffer["rgb"].shape[1], seq) - 1
                    actions = actions[0, idx, :, :].unsqueeze(0).unsqueeze(0)
                    values = (
                        predictions["value"]["improve"][0, idx, :, :]
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    value, action = self.model.MO.value_net._predict(values, actions)

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

            env.to_wandb()
            env.close()

    def epoch(self, epoch):

        # self.rollout()
        for batch in tqdm(self.loader, total=len(self.loader), desc="epoch"):
            self.step(batch)
        self.save()

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
            n_epochs = self.cfg.training.num_steps // len(self.loader)
            for _ in tqdm(range(n_epochs), desc="steps"):
                self.epoch(epoch)
                self.log({"time/epoch": epoch}, self.nstep)
                epoch += 1
                # self.Logger.log()
