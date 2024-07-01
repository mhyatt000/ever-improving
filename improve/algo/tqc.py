import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict as TD
from tensordict.nn import TensorDictModule as TDM
from tensordict.nn import TensorDictSequential as TDS

from improve.wrapper import dict_util as du

from .base import Algo


class TQC(Algo):

    def transform_batch(self, batch):
        del batch["observation"]["simpler-img"]
        batch = TD(batch, batch_size=self.cfg.data.bs_per_gpu) 
        return batch

    def loss(self, x):
        # print(x)
        x["reward"] = x["reward"].unsqueeze(-1)

        # TODO this doesnt use TD1 value
        loss = {
            "actor": F.mse_loss(x["output", "action"], x["action"]),
            "critic": F.mse_loss(x["output", "value"], x["reward"]),
        }
        loss["total"] = loss["actor"] + loss["critic"]
        return loss
