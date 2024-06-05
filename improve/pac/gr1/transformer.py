import json
import math
import os
import os.path as osp
from datetime import timedelta
from time import time

import clip
import hydra
import improve
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs,
                              InitProcessGroupKwargs)
from flamingo_pytorch import PerceiverResampler
from improve.data.flex import HDF5IterDataset
from improve.pac.gr1.util.optim import async_step
from improve.pac.gr1.util.prefetch import DataPrefetcher
from improve.pac.gr1.util.transform import PreProcess
from improve.wrapper import dict_util as du
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from torch.profiler import (ProfilerActivity, profile, record_function,
                            tensorboard_trace_handler)
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Model, get_cosine_schedule_with_warmup

import models.vision_transformer as vits
from models.gr1 import GR1
from models.transformer_utils import get_2d_sincos_pos_embed
from models.vision_transformer import Block
from util.loss import masked_loss


def init_gpt2_model(self, hidden_size, **kwargs):
    config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
    return GPT2Model(config)


def create_causal_mask(seq_len):
    """Creates a causal mask for a transformer decoder.
    ensures each token in the sequence can only attend to earlier positions
    fill `-inf` for the mask positions
    A += attention_mask
    A = F.softmax(A, dim=-1) and softmax of -inf is 0 which masks them
    """

    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


@hydra.main(
    config_path=osp.dirname(__file__), config_name="gr1_config", version_base="1.3.2"
)
def main(cfg):

    cn_gpt = OC.to_container(cfg.model.gpt_kwargs, resolve=True)
    gpt = init_gpt2_model(GR1, cfg.model.embed_dim, **cn_gpt)

    print(gpt)

    batch_size = 1  # for now
    seq_len = 10  # Assuming your input sequence length is 10

    # Assuming your hidden size is 384 (from the GPT2 model)
    hidden_size = 384

    stacked_inputs = torch.rand((1, seq_len, hidden_size))

    # Create a causal mask for stacked_attn_mask
    stacked_attn_mask = create_causal_mask(seq_len)

    print(stacked_inputs.shape)
    print(stacked_attn_mask.shape)
    print(stacked_attn_mask)
    quit()

    outs = gpt(inputs_embeds=stacked_inputs, attention_mask=stacked_attn_mask)
    print(outs["last_hidden_state"].reshape(batch_size, seq_len, -1, hidden_size))

    print(outs)


if __name__ == "__main__":
    main()
