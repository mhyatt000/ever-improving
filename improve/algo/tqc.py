import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict as TD
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential as TDS

from improve.wrapper import dict_util as du

from .base import Algo
from .target import TargetNet

from stable_baselines3.common.utils import get_parameters_by_name, polyak_update


class TQC(TargetNet):

    def __init__(self, fabric, model, loader, optimizer, scheduler, cfg):
        super().__init__(fabric, model, loader, optimizer, scheduler, cfg)

        self.gamma = 0.99  # cfg.algo.gamma

    def transform_batch(self, batch):
        # del batch["observation"]["simpler-img"]
        batch = TD(batch, batch_size=self.cfg.data.bs_per_gpu)
        return batch

    def loss(self, x):

        # TODO this is simplified loss rn
        loss = {
            "actor": 0,  # F.mse_loss(x["output", "action"], x["action"]),
            "critic": self.critic_loss(x),
        }
        loss["total"] = loss["actor"] + loss["critic"]
        return loss

    def critic_loss(self, x):
        """naive critic loss
        TODOS
        - use entropy term
        - use greedy actor as target***
        - use min of multiple critics as target
        """

        with torch.no_grad():
            future = x["target", "value"].detach().roll(-1, dims=1)
            target = (
                x["reward"].unsqueeze(-1)
                + +(1 - x["terminated"].unsqueeze(-1)) * self.gamma * future
            )

        # fake = x["reward"].unsqueeze(-1) + +(1) * self.gamma * future

        loss = F.mse_loss(x["base", "value"], target)

        self.fabric.log_dict(
            {
                "stats/value_mean": x["base", "value"].mean(),
                "stats/value_std": x["base", "value"].std(),
            }
        )
        # Compute critic loss for multiple critics
        # loss = 0.5 * sum( F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

        return loss
