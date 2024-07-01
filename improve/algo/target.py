from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict as TD
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential as TDS

from improve.wrapper import dict_util as du

from .base import Algo


class TargetNet(Algo):
    """Training with a target network"""

    def __init__(self, fabric, model, loader, optimizer, scheduler, cfg):
        assert "target" in model.keys()
        super().__init__(fabric, model, loader, optimizer, scheduler, cfg)

        self.tau = 0.005  # cfg.algo.tau

    def transform_batch(self, batch):
        return batch

    def maybe_backward(self, loss):
        if True:
            self.fabric.backward(loss)
            self.fabric.clip_gradients(self.model["base"], self.optimizer, max_norm=2.0)
            self.optimizer.step()
            self.scheduler.step()

            polyak_update(
                self.model["base"].parameters(),
                self.model["target"].parameters(),
                self.tau,
            )
            # Copy running stats, see GH issue #996
            # polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        else:
            # accumulate gradient ... not ready yet
            # see Algo
            raise NotImplementedError

    def step(self, batch):

        for k, m in self.model.items():
            m.train()
        self.optimizer.zero_grad()

        batch = self.transform_batch(batch)

        for k, m in self.model.items():
            batch = m(batch)

        loss = self.loss(batch)
        self.maybe_backward(loss["total"])

        # housekeeping
        self.fabric.log_dict(
            {
                "loss": loss,
                "train/lr": self.optimizer.param_groups[0]["lr"],
            },
            self.nstep,
        )
        self.nstep += 1
