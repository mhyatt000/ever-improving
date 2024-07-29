"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small --data_dir=...
"""

import improve.wrapper.dict_util as du
from dataclasses import dataclass
from typing import Any, Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import qax
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags, logging
from flax import struct
from octo.data.dataset import make_single_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (TrainState, freeze_weights, merge_params,
                                    process_text)
from octo.utils.typing import Config, Data, Params, PRNGKey


import os.path as osp

from pprint import pprint
import lora_octo
import simpler_env as simpler
import lorax


from dataclasses import asdict, dataclass


@dataclass
class MyConfig:
    pretrained_path: Optional[str] = None
    data_dir: Optional[str] = None
    save_dir: Optional[str] = None
    freeze_transformer: bool = False

    seed: int = 0
    batch_size: int = 64 # 8  # 16 is slow
    grad_acc: Optional[int] = 8  # really should be ≈128
    grad_clip: Optional[int] = 2

    train_steps: int = int(3e5)
    sweep_id: str = "lora"


cfg = MyConfig()


@struct.dataclass
class MyTrainState:
    rng: PRNGKey
    model: OctoModel
    params: Any
    step: int
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, rng):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, params=self.params
        )
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            model=self.model,
            params=new_params,
            opt_state=new_opt_state,
            rng=rng,
        )


def main(_):

    print("Using wandb")
    wrun = wandb.init(
        project="lora",
        dir= osp.join(osp.expanduser('~'),'improve_logs'), # cfg.callback.log_path,
        job_type="train",
        # sync_tensorboard=True,
        monitor_gym=True,
        config=asdict(cfg) , # OC.to_container(cfg, resolve=True),
    )
    wandb.config.update({"name": wrun.name})



    assert (
        cfg.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")
    rng = jax.random.PRNGKey(cfg.seed)

    # setup wandb for logging
    # wandb.init(name="finetune_aloha", project="octo")

    # load pre-trained model
    # logging.info("Loading pre-trained model...")
    model_type = "octo-small"
    model_type = f"hf://rail-berkeley/{model_type}"
    pretrained_model = OctoModel.load_pretrained(model_type)

    # from demo
    # pretrained_model = OctoModel.load_pretrained(cfg.pretrained_path)

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    # logging.info("Loading finetuning dataset...")

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    # @jax.jit
    def process_batch(batch, text_processor):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch



    dataset = iter(lora_octo.octo_dataset(cfg.batch_size))
    batch = next(dataset)

    task = "widowx_put_eggplant_in_basket"
    env = simpler.make(task)
    descs = [env.get_language_instruction()] * cfg.batch_size

    batch["task"]["language_instruction"] = descs
    # example_batch = process_batch(batch, text_processor)
    lang = pretrained_model.create_tasks(texts=descs)["language_instruction"]
    batch["task"] = {"language_instruction": lang}
    example_batch = batch

    # if self.automatic_task_creation:
    # print(tasks)
    # pprint(jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), lang))

    example_batch_spec = jax.tree.map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    print("example_batch_spec")

    pprint(example_batch_spec)

    # train_data_iter = map(process_batch, train_data_iter)
    # example_batch = next(train_data_iter)
    # example_batch = { "action": np.zeros((8, 5, 7), dtype="float32"), "observation": { "image_primary": np.zeros((8, 2, 256, 256, 3), dtype="uint8"), "pad_mask": np.zeros((8, 2), dtype="int32"), }, "task": { "language_instruction": { "attention_mask": np.zeros((8, 16), dtype="int64"), "input_ids": np.zeros((8, 16), dtype="int64"), } }, }

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    ###
    """
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    # Fully override the old action head with a new one (for smaller changes, you can use update_module_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        pred_horizon=5,  # 50 for aloha but thats too much for simpler
        action_dim=14,
        readout_key="readout_action",
    )
    """

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    # logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        # dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model


    def decision_fn(path, param):

        path = [str(p.key) for p in path]
        joined = osp.join(*path)
        dim = 32

        if "task_tokenizers_language" in path or "hf_model" in path:
            # print(f"Freezing {joined}")
            return lorax.LORA_FREEZE

        if ("BlockTransformer_0" in path or "Transformer_0" in path) and (
            "MlpBlock_0" in path
        ):  # no MHA LoRA
            print(f"Using LoRA with dim={dim} for param {joined} | {param.shape}")
            return dim

        if ("heads_action" and "diffusion_model" in path) and (
            "Dense_0" in path or "Dense_1" in path
        ):
            print(f"Using LoRA with dim={dim} for param {joined}")
            return dim

        print(joined)
        return lorax.LORA_FREEZE

        if False:
            print(f"Fully finetuning param {path}")
            return lorax.LORA_FULL

    lora_spec = lorax.simple_spec(
        model.params, decision_fn=decision_fn, tune_vectors=True
    )
    print(lora_spec)

    def LoRAify(obj):
        """
        Recursively wraps the __call__ methods of nn.Module attributes in lorax.lora(__call__).
        Args:
            obj: An object that is either an nn.Module or has attributes that are nn.Module.
        """

        if isinstance(obj, nn.Module):
            for attr_name in dir(obj):
                attr = getattr(obj, attr_name)
                if isinstance(attr, nn.Module):
                    # Recursively wrap nested nn.Modules
                    LoRAify(attr)
                elif callable(attr):
                    # Check if the attribute is the __call__ method
                    if attr_name == "__call__":
                        wrapped_call = lorax.lora(attr)
                        setattr(obj, attr_name, wrapped_call)

        else:
            for attr_name in dir(obj):
                attr = getattr(obj, attr_name)
                if isinstance(attr, nn.Module):
                    LoRAify(attr)
                elif callable(attr):
                    # Check if the attribute is the __call__ method
                    if attr_name == "__call__":
                        wrapped_call = lorax.lora(attr)
                        setattr(obj, attr_name, wrapped_call)

    # model = model.bind({"params": params}, rngs={"dropout": rng})
    # model.module( example_batch["observation"], example_batch["task"], example_batch["observation"]["pad_mask"], train=False,)
    # LoRAify(model)
    # model.module.octo_transformer.__call__ = lorax.lora(model.module.octo_transformer.__call__ )
    # model.module.heads['action'].__call__ = lorax.lora(model.module.heads['action'].__call__ )

    # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
    # which had a spec value other than LORA_FULL or LORA_FREEZE
    lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(0))
    # target_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(1))

    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(
        learning_rate=learning_rate, weight_decay=1e-4, mu_dtype=jnp.bfloat16
    )
    if cfg.grad_acc:
        tx = optax.MultiSteps(tx, cfg.grad_acc)
    if cfg.grad_clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), tx)

    tx = lorax.wrap_optimizer(tx, lora_spec)
    opt_state = tx.init(lora_params)

    train_state = MyTrainState(
        rng=rng,
        model=model,
        params=lora_params,
        step=0,
        opt_state=opt_state,
        tx=tx,
    )

    # `wrap_optimizer` uses the spec to freeze the appropriate subset of parameters.
    # The frozen parameters won't have optimizer states etc created for them
    tx = lorax.wrap_optimizer(tx, lora_spec)
    opt_state = tx.init(lora_params)

    """
    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if cfg.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )
    """

    # define loss function and train step
    @lorax.lora
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})

        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(range(cfg.train_steps), total=cfg.train_steps, dynamic_ncols=True):
        batch = next(dataset)
        batch["task"] = {"language_instruction": lang}
        # batch = example_batch
        # batch = process_batch(batch, text_processor)

        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            print(update_info)
            update_info = jax.device_get(update_info)
            wandb.log( du.flatten({"training": update_info}, delim="/"), step=i)

        # if (i + 1) % 1000 == 0:
        # save checkpoint
        # train_state.model.save_pretrained(step=i, checkpoint_path=cfg.save_dir)


if __name__ == "__main__":
    app.run(main)
