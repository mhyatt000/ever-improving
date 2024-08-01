import os
import os.path as osp
import warnings
from dataclasses import asdict, dataclass, field
from functools import partial
from pprint import pprint
from typing import Any, Dict, List, Optional

import hydra
import jax
import jax.numpy as jnp
import lorax
import numpy as np
import octo
import optax
import simpler_env as simpler
import tensorflow as tf
import torch
import wandb
import webdataset as wds
from flax import struct
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import RolloutVisualizationCallback
from octo.utils.train_utils import (Timer, TrainState, check_config_diff,
                                    create_optimizer, format_name_with_config,
                                    merge_params, process_text)
from octo.utils.typing import Config, Data, Params, PRNGKey
from omegaconf import OmegaConf as OC
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm
from transformers import FlaxGPT2LMHeadModel

import improve
import improve.wrapper.dict_util as du
from improve.data.lorax import find_tarballs, mk_dataset, preprocess
from improve.env import make_env, make_envs
from improve.fm.batch_octo import BatchedOctoInference
from improve.fm.cache import load_task_embedding

# prevent tensorflow from using GPU memory since it's only used for data loading
tf.config.set_visible_devices([], "GPU")


@dataclass
class MyConfig:
    num_steps: int = int(3e5)
    seed: int = 0
    batch_size: int = 8  # 16 is slow
    grad_acc: Optional[int] = 4  # really should be ≈128
    grad_clip: Optional[int] = 2

    sweep_id: str = "lora"

    # foundation: Dict[str, Any] = field(default_factory=lambda: {'name': 'octo-base', 'ckpt': None, 'task': 'widowx_put_eggplant_in_basket', 'noact': [-1, -2, -3, -4], 'strategy': 'clip', 'residual_scale': 1.0, 'batch_size': 8})
    # obs_mode: Dict[str, Any] = field(default_factory=lambda: {'name': 'oracle-central', 'mode': 'rgb', 'obs_keys': ['obj-wrt-eef', 'agent_qpos-sin', 'agent_qpos-cos', 'agent_qvel', 'eef-pose', 'agent_partial-action', 'simpler-img']})
    # task: str = '${.foundation.task}'
    # bonus: bool = False
    # kind: str = 'sb3'
    # downscale: int = 7
    # device: Optional[str] = None
    # obs_keys: str = '${env.obs_mode.obs_keys}'
    # goal: Dict[str, Any] = field(default_factory=lambda: {'use': True, 'key': 'simpler-img', 'cls': '${r_typeof:improve.wrapper.GoalEnvWrapper}'})
    # residual_scale: int = 1
    # scale_strategy: str = 'clip'
    # action_mask_dims: Optional[Any] = None
    # use_original_space: bool = False
    # seed: Dict[str, Any] = field(default_factory=lambda: {'force': False, 'value': '${job.seed}', 'seeds': None})
    # reward: str = 'sparse'
    # max_episode_steps: int = 60
    # no_quaternion: bool = False
    # reach: bool = False
    # fm_loc: str = 'central'
    record: bool = True


cfg = MyConfig()


def run(train_state, train_data_iter, train_step, lang, rollout_callback):

    timer = Timer()
    for i in tqdm(
        range(0, int(cfg.num_steps)),
        total=int(cfg.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            # batch = next(train_data_iter)
            # BUG: Dummy Data
            batch = {
                "action": np.zeros((8, 5, 7), dtype="float32"),
                "observation": {
                    "image_primary": np.zeros((8, 2, 256, 256, 3), dtype="uint8"),
                    "pad_mask": np.zeros((8, 2), dtype="int32"),
                },
                "task": {
                    "language_instruction": {
                        "attention_mask": np.zeros((8, 16), dtype="int64"),
                        "input_ids": np.zeros((8, 16), dtype="int64"),
                    }
                },
            }
            batch["task"] = {"language_instruction": lang}

            # print("batch shape:", batch)
            # batch_spec = jax.tree.map(
            #     lambda arr: (arr.shape, str(arr.dtype)), batch
            # )
            # breakpoint()
            # {'action': ((8, 5, 7), 'float32'), 'observation': {'image_primary': ((8, 2, 256, 256, 3), 'uint8'), 'pad_mask': ((8, 2), 'int32')}, 'task': {'language_instruction': {'attention_mask': ((8, 16), 'int64'), 'input_ids': ((8, 16), 'int64')}}}

            # print(batch.items())
            # print(batch["action"])
            # action': ((8, 5, 7)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)
            print("-----------\nIn Run function")
            print(update_info["actions"].shape)
            # print(update_info)
            flattened_actions = update_info["actions"].reshape(-1, 7)

            loggedInfo = {x: update_info[x] for x in update_info if x != "actions"}

            wandb.log(loggedInfo, step=i)
            # wandb.log(update_info, step=i)
            print("wandb logged for loggedInfo")

            # Log histograms for each component
            components = ["x", "y", "z", "yaw", "pitch", "roll", "gripper_state"]

            for j, component in enumerate(components):
                wandb.log(
                    {
                        f"prediction/actions/{component}": wandb.Histogram(
                            flattened_actions[:, j]
                        )
                    },
                    step=i,
                )
                print("wandb logged for", component)

        timer.tock("total")

        cfg.eval_interval = 10
        if (i + 1) % cfg.eval_interval == 0:
            print("Evaluating...")
            if rollout_callback is not None:
                with timer("rollout"):
                    rollout_metrics = rollout_callback(train_state, i + 1)
                    wandb.log(rollout_metrics, step=i)

        """
        if (i + 1) % cfg.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i + 1) % cfg.eval_interval == 0:
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            with timer("visualize"):
                viz_metrics = viz_callback(train_state, i + 1)
                wandb_log(viz_metrics, step=i)

            if rollout_callback is not None:
                with timer("rollout"):
                    rollout_metrics = rollout_callback(train_state, i + 1)
                    wandb_log(rollout_metrics, step=i)

        if (i + 1) % cfg.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)
        """


def tuple2dict(x):
    return {
        "obs": jax.tree.map(lambda a: jnp.array(a), x[0]),
        "next_obs": jax.tree.map(lambda a: jnp.array(a), x[1]),
        "action": jnp.array(x[2]),
        "reward": jnp.array(x[3]),
        "done": jnp.array(x[4]),
        # "info": x[5],
    }


def make_values(x: dict):
    rew = x["reward"].tolist()

    for i in range(len(rew) - 2, -1, -1):
        rew[i] = 0.99 * rew[i + 1]
    x["value"] = jnp.expand_dims(jnp.array(rew), axis=-1)

    return x


def future_actions(x: dict):
    return mk_horizon(x, 5, "action")


def mk_horizon(x: dict, horizon, key):

    things = x[key]
    n = things.shape[0]
    dims = list(things.shape[1:])

    # Create a new array of zeros with shape (n, horizon, 7)
    new = jnp.zeros([n, horizon] + dims)

    # Use roll to shift the array and vmap to apply it over a range of shifts
    def get_future(idx):
        return jnp.stack(
            [jnp.roll(things, -shift, axis=0)[idx] for shift in range(horizon)], axis=0
        )

    get_future_vmap = jax.vmap(get_future, in_axes=(0,))
    new = get_future_vmap(jnp.arange(n))

    # Handle edge cases where we roll beyond the length of the array
    mask = jnp.arange(n).reshape(-1, 1) + jnp.arange(horizon) < n
    # new = jnp.where(mask[:, :, None], new, actions[-1])

    # should predict 0s when done
    new = jnp.where(mask[:, :, None], new, jnp.zeros_like(things[-1]))

    x[key] = new
    return x


"""
def split_sample(x, n=8):
    l = len(x["done"])
    out = []
    for i in range(l // n):
        stop, start = -i * n, -(i + 1) * n
        out.append(jax.tree.map(lambda a: a[start:stop], x))
    yield out
"""


def split_sample(x, n=8):
    l = len(x["done"])
    out = []
    for i in range(l):
        out.append(jax.tree.map(lambda a: a[i], x))
    return out


def dict_collate(samples, combine_tensors=True, combine_scalars=True):
    """Take a collection of samples (dictionaries) and create a batch.

    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.

    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict

    """
    assert isinstance(samples[0], (dict)), type(samples[0])

    def list2dict(lst):
        if isinstance(lst[0], dict):
            keys = [k for k in lst[0].keys() if "__" not in k]
            result = {k: list2dict([d[k] for d in lst]) for k in keys}
            return result
        else:
            if lst[0] is None:
                return None
            if isinstance(lst[0], torch.Tensor):
                return torch.stack(lst) if not lst[0] is None else None

            return jnp.stack(lst) if not lst[0] is None else None

    return list2dict(samples)


def _resize_image(image: np.ndarray) -> np.ndarray:
    image = tf.image.resize(
        image,
        size=(256, 256),
        method="lanczos3",
        antialias=True,
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
    return image


def prepare_octo(x):
    x["observation"] = {}
    x["observation"]["image_primary"] = _resize_image(x["obs"]["simpler-img"])
    x["action"] = x["action"][0]
    x["value"] = x["value"][0] if "value" in x else x['value']
    del x["obs"]

    x.pop("__key__", None)
    del x["next_obs"]
    del x["done"]
    return x


def get_mask(x):
    mask_val = sum(x["reward"])
    # mask_val = mask_val if mask_val != 0 else -0.1
    x["observation"]["pad_mask"] = jnp.array([0, 0])
    x["observation"]["pad_mask"] = jnp.array([1, 1])
    del x["reward"]
    return x


def get_task(x, task: str = "widowx_put_eggplant_in_basket"):
    embed = jnp.array(load_task_embedding(task)[0])  # remove time dimension for now
    x["task"] = {"language_instruction": embed}
    return x


def is_batch(x, batch_size):
    # print(x.keys())
    # pprint(jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), x))
    x = {k: v for k, v in x.items() if "__" not in k}
    sz = jax.tree.reduce(lambda s, arr: s and arr.shape[0] == batch_size, x, True)
    return sz


def octo_dataset(batch_size):
    HOME = os.environ["HOME"]
    dataset = ["sunny-eon-12"]
    exp_root = [osp.join(HOME, "improve_logs", x) for x in dataset]

    # dnames = [osp.join(e, "eval") for e in exp_root]
    dnames = [osp.join(e, "train") for e in exp_root]
    fnames = list(find_tarballs(dnames))
    print(fnames)
    fnames = [f for f in fnames if "pt" in f]

    dataset = wds.DataPipeline(
        # wds.SimpleShardList(fnames),
        # use resampled shards if you want to loop the dataset
        wds.ResampledShards(fnames, deterministic=True),
        # at this point we have an iterator over all the shards
        # wds.shuffle(100), # shuffles the shards
        #
        # add wds.split_by_node here if you are using multiple nodes
        wds.split_by_node,
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(),
        # this shuffles the samples in memory
        wds.shuffle(500),  # shuffles the samples... too much shuffle will kill
        # this decodes the images and json
        wds.decode(),  # BUG: conflicts with having multiple workers, but we want multiple workers
        # wds.to_tuple("png", "json"),
        wds.map(preprocess),
        wds.map(tuple2dict),  # not needed if already dict
        wds.map(lambda x: mk_horizon(x, 5, "action")),
        wds.map(make_values),
        wds.select(lambda x: x['reward'].sum() > 0 ), # only successful episode
        wds.map(lambda x: mk_horizon(x, 5, "value")),
        wds.map(split_sample),
        wds.filters.unlisted(),  # (split_sample)
        # wds.to_tuple(),
        # For IterableDataset objects, the batching needs to happen in the dataset.
        wds.batched(2, collation_fn=dict_collate),
        wds.map(prepare_octo),
        wds.map(get_mask),
        wds.map(get_task),
        wds.batched(batch_size, collation_fn=dict_collate),
        wds.select(lambda x: is_batch(x, batch_size)),
    )

    loader = wds.WebLoader(
        dataset,
        # batch_size=None,
        num_workers=8,
        prefetch_factor=2,
        # pin_memory=True,
        # persistent_workers=True,  # Persistent workers
    )
    return dataset


@struct.dataclass
class MyTrainState:
    rng: PRNGKey
    model: OctoModel
    params: Any
    step: int
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, rng):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            model=self.model,
            params=new_params,
            opt_state=new_opt_state,
            rng=rng,
        )


# @hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main():

    print("Using wandb")
    wrun = wandb.init(
        project="awac",
        dir=osp.join(osp.expanduser("~"), "improve_logs"),  # cfg.callback.log_path,
        job_type="train",
        # sync_tensorboard=True,
        monitor_gym=True,
        config=asdict(cfg),  # OC.to_container(cfg, resolve=True),
    )
    wandb.config.update({"name": wrun.name})

    # try AWDI
    # add value readout

    # 1. load the dataset
    dataset = octo_dataset()

    # dataset = dataset.map(process)
    example_batch = next(iter(dataset))
    # print(example_batch)
    # print(example_batch.keys())
    # example_batch.pop("info")

    example_batch_spec = jax.tree.map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    print("example_batch_spec")
    pprint(example_batch_spec)
    # print(example_batch)

    """
      File "/home/mhyatt000/cs/octo/octo/model/octo_model.py", line 375, in <lambda>
        example_batch = jax.tree.map(lambda x: x[:1], example_batch)
    IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
    """

    # should load sequences of image and text

    rng = jax.random.PRNGKey(cfg.seed)

    # 2. build the model

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    model_type = "octo-small"
    model_type = f"hf://rail-berkeley/{model_type}"
    pretrained = OctoModel.load_pretrained(model_type)

    pprint(pretrained.config, indent=2, compact=True)
    # quit()

    if cfg.batch_size > 1:
        example_batch = jax.tree.map(
            lambda x: np.concatenate([x] * cfg.batch_size), pretrained.example_batch
        )
    pprint(jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), example_batch))

    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(
        pretrained.config,
        example_batch,
        pretrained.text_processor,
        verbose=True,
        # rng=init_rng,
    )
    # breakpoint()

    merged_params = merge_params(model.params, pretrained.params)
    model = model.replace(params=merged_params)
    del pretrained

    task = "widowx_put_eggplant_in_basket"
    env = simpler.make(task)
    descs = [env.get_language_instruction()] * cfg.batch_size

    # if self.automatic_task_creation:
    lang = model.create_tasks(texts=descs)["language_instruction"]
    # print(tasks)
    pprint(jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), lang))

    # print(isinstance(lang, jax.Array))
    # print(type(lang))

    # 3. create the optimizer

    # you can insert lora here

    kwargs = {
        "learning_rate": 3e-4,
        "clip_gradient": None,
        "frozen_keys": None,
        "grad_accumulation_steps": None,
        "mu_dtype": jnp.bfloat16,
    }

    #
    # TODO add Octo optimizations
    #

    grad_accumulation_steps = kwargs.pop("grad_accumulation_steps", None)

    # tx = optax.adamw(mu_dtype=jnp.bfloat16, **kwargs, mask=wd_mask)

    # This function defines a spec which tells lorax how each parameter should be handled
    """
    def decision_fn(path, param):
        if "embedding" in path:
            print(f"Fully finetuning param {path}")
            return lorax.LORA_FULL
        dim = 32
        print(f"Using LoRA with dim={dim} for param {path}")
        return dim
    """

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
            print(f"Using LoRA with dim={dim} for param {joined}")
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

    # Create a pytree with the same shape as params indicating how each parameter should be handled
    # Each leaf will be given one of the following values:
    # - LORA_FULL: The parameter will be fully finetuned
    # - LORA_FREEZE: The parameter will be frozen
    # - k > 0: The parameter will be LoRA tuned with a rank k update

    # Simple_spec is a helper to do this, but you can also create the label pytree yourself
    lora_spec = lorax.simple_spec(
        model.params, decision_fn=decision_fn, tune_vectors=True
    )

    # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
    # which had a spec value other than LORA_FULL or LORA_FREEZE
    lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(0))
    # target_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(1))

    tx = optax.adamw(learning_rate=1e-4, weight_decay=1e-4, mu_dtype=jnp.bfloat16)
    if cfg.grad_acc:
        tx = optax.MultiSteps(tx, cfg.grad_acc)
    if cfg.grad_clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), tx)

    # `wrap_optimizer` uses the spec to freeze the appropriate subset of parameters.
    # The frozen parameters won't have optimizer states etc created for them
    tx = lorax.wrap_optimizer(tx, lora_spec)
    opt_state = tx.init(lora_params)

    # lorax.lora wraps a callable so that the arguments can be lorax.LoraWeight
    # instances. (It's actually just an alias for qax.use_implicit_args, so
    # the wrapped function can handle other qax types as well)
    lora_model = lorax.lora(model)

    """
    print(type(lora_model.params ))
    print(type(lora_params))
    def func(x,y):
        print(x)
        print(y)
        print()
    (du.apply_both(lora_model.params, lora_params, func))
    quit()
    """

    # tx, lr_callable, param_norm_callable = create_optimizer(params, **kwargs)
    # train_state = TrainState.create(model=lora_model, tx=tx, rng=rng)
    train_state = MyTrainState(
        rng=rng,
        model=lora_model,
        params=lora_params,
        step=0,
        opt_state=opt_state,
        tx=tx,
    )

    print(type(train_state))

    # lora_transformer = lorax.lora(model.module.octo_transformer)
    # lora_head = lorax.lora(model.module.heads['action'].loss)

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

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    # @jax.jit
    # @partial(jax.jit, in_shardings=[replicated_sharding, dp_sharding])
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, dropout_rng, train=True
        )

        # Gradient Metrics (FIXME: Does the finetuner need these?) ###
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.params)
        update_norm = optax.global_norm(updates)

        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                # "param_norm": param_norm_callable(state.model.params),
                # "learning_rate": lr_callable(state.step),
            }
        )
        # End Debug Metrics #

        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # 4. run the training loop

    # TODO: CREATE ENVIRONMENT HERE TO PASS TO RUN
    cfg.env.record = True
    log_dir = osp.join(cfg.callback.log_path, wrun.name) if cfg.job.wandb.use else None

    if cfg.job.seed is not None:
        set_random_seed(cfg.job.seed)

    eval_only = not cfg.train.use_train

    # BUG Check that this is correct
    env, eval_env = make_envs(
        cfg,
        log_dir,
        eval_only=eval_only,
        num_envs=8,
    )
    # breakpoint()

    rollout_callback = RolloutVisualizationCallback(
        env=env,
        text_processor=model.text_processor,
        history_length=2,
        model_pred_horizon=model.config["model"]["heads"]["action"]["kwargs"].get(
            "pred_horizon", 1
        ),
    )

    run(train_state, iter(dataset), train_step, lang, rollout_callback)

    print("ready for lora")

    quit()

    # No changes are necessary to the loss function apart from using the wrapped model
    def loss_fn(lora_params, batch):
        input_ids = batch[:, :-1]

        # The call signature of the wrapped model is unchanged from the original HuggingFace model
        logits = lora_model(input_ids, params=lora_params).logits

        logprobs = jax.nn.log_softmax(logits)
        target_logprobs = jnp.take_along_axis(logprobs, batch[:, 1:, None], axis=-1)
        return -jnp.mean(target_logprobs)

    # The update function also doesn't need to be modified other than
    # using the wrapped optimizer
    @jax.jit
    def update_fn(lora_params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(lora_params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=lora_params)

        new_params = optax.apply_updates(lora_params, updates)
        return new_params, new_opt_state, loss

    # Train on a dummy batch to show that we can fit the model to stuff
    example_data = jax.random.randint(jax.random.PRNGKey(0), (4, 128), 0, 50257)
    bar = tqdm(range(int(2e3)))
    for _ in bar:
        lora_params, opt_state, loss = update_fn(lora_params, opt_state, example_data)
        bar.set_description(f"loss: {loss:.4f}")
        bar.update(1)

    final_predictions = lora_model(example_data, params=lora_params).logits

    # Now let's merge the loras back into the original parameters to get
    # finetuned parameters we can use with no extra compute
    merged_params = lorax.merge_params(lora_params)

    orig_model_predictions = model(example_data, params=merged_params).logits

    gap = jnp.max(jnp.abs(final_predictions - orig_model_predictions))
    print(final_predictions[0])
    print(orig_model_predictions[0])
    print(f"Max prediction gap: {gap:.3e}")


if __name__ == "__main__":
    main()
