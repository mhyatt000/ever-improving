# import os.path as osp
from dataclasses import asdict, dataclass
from functools import partial
from pprint import pprint
from typing import Any, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import gymnasium as gym
import improve.wrapper.dict_util as du
import jax
import jax.numpy as jnp
# import lorax
import numpy as np
import optax
# import qax
import simpler_env as simpler
import tensorflow as tf
from absl import app, flags, logging
from flax import struct
# from improve import cn, lora_octo
from improve.env.action_rescale import ActionRescaler
from improve.offline.awac import mk_model_step, mk_octo_adv_loss
from improve.offline.critic_heads import MSECriticHead
from improve.util.config import default
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from octo.data.dataset import make_single_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.model.components.action_heads import L1ActionHead, MSEActionHead
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (Timer, TrainState, freeze_weights,
                                    merge_params, process_text)
from octo.utils.typing import Config, Data, Params, PRNGKey
from tqdm import tqdm

import wandb

"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).
To run this example, first download and extract the dataset from here: 
    https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip
python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small --data_dir=...
"""


@dataclass
class MyConfig:
    pretrained_path: Optional[str] = None
    data_dir: Optional[str] = None
    save_dir: Optional[str] = None
    freeze_transformer: bool = False

    seed: int = 0

    gpus: int = jax.device_count()
    batch_size: int = 64 * gpus

    inference_size: int = 5
    # batch_size: int = 64
    grad_acc: Optional[int] = 8  # total = 64 * 8 = 512
    grad_clip: Optional[int] = 2

    train_steps: int = int(5e4)  # int(3e5)
    sweep_id: str = "lora"

    env: cn.Env = default(cn.Env())
    task: str = "widowx_put_eggplant_in_basket"


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


def isimg(o):
    if isinstance(o, np.ndarray) and o.ndim == 3:
        return o.shape[-1] in [1, 3, 4]
    return False


from improve.fm.batch_octo import BatchedActionEnsembler
from transforms3d.euler import euler2axangle


class OXE2SimplerProcesser:

    def __init__(
        self,
        dataset_id="bridge_dataset",
        policy_setup="widowx_bridge",
        batch_size=cfg.batch_size,
    ):

        self.action_scale = 1

        if dataset_id == "bridge_dataset":
            self.action_mean = np.array(
                [
                    0.00021161,
                    0.00012614,
                    -0.00017022,
                    -0.00015062,
                    -0.00023831,
                    0.00025646,
                    0.0,
                ]
            )
            self.action_std = np.array(
                [
                    0.00963721,
                    0.0135066,
                    0.01251861,
                    0.02806791,
                    0.03016905,
                    0.07632624,
                    1.0,
                ]
            )
        elif dataset_id == "fractal20220817_data":
            self.action_mean = np.array(
                [
                    0.00696389,
                    0.00627008,
                    -0.01263256,
                    0.04330839,
                    -0.00570499,
                    0.00089247,
                    0.0,
                ]
            )
            self.action_std = np.array(
                [
                    0.06925472,
                    0.06019009,
                    0.07354742,
                    0.15605888,
                    0.1316399,
                    0.14593437,
                    1.0,
                ]
            )
        else:
            raise NotImplementedError(
                f"{dataset_id} not supported yet for custom octo model checkpoints."
            )
        self.automatic_task_creation = False

        self.batch_size = batch_size

        self.horizon = 2  # state history
        self.pred_action_horizon = 4
        self.exec_horizon = 1  # RHC receding horizon control

        # from policy_setup
        self.policy_setup = policy_setup
        self.action_ensemble = True
        self.action_ensemble_temp = 0.0
        self.sticky_gripper_num_repeat = 1  # 15 for google robot

        if self.action_ensemble:
            self.action_ensembler = BatchedActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp, self.batch_size
            )
        else:
            self.action_ensembler = None

        from collections import deque

        self.image_history = deque(maxlen=self.horizon)

    def reset(self):

        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = np.full((self.batch_size,), False)
        self.gripper_action_repeat = np.full((self.batch_size,), 0)
        self.sticky_gripper_action = np.full((self.batch_size,), 0.0)
        # parent removed this ... maybe its for the alternate method?
        # self.gripper_is_closed = False
        self.previous_gripper_action = np.full((self.batch_size,), np.nan)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        ax = 1  # 0 if self.batch_size == 1 else 1
        images = np.stack(self.image_history, axis=ax)  # stack for OctoInference
        horizon = len(self.image_history)
        # note: this should be of float type, not a bool type
        pad_mask = np.ones(horizon, dtype=np.float64)
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this should be of float type, not a bool type
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask

    def pre_step(self, obs):
        # resized in env wrapper
        # image = self._resize_image(image)

        self._add_image_to_history(obs["observation"]["image_primary"])
        images, pad_mask = self._obtain_image_history_and_mask()
        pad_mask = np.repeat(pad_mask[None, :], self.batch_size, axis=0)
        obs["observation"]["image_primary"] = images
        obs["observation"]["pad_mask"] = pad_mask

        return obs

    def __call__(self, nactions):

        # 1. ensemble
        # 2. unnormalize
        # 3. maybe scale
        # 4. rpy to axangle
        # 5. sticky gripper

        if self.action_ensemble:
            nactions = self.action_ensembler.ensemble_action(nactions)

        # octo predicts normalized actions

        raw_actions = nactions * self.action_std[None] + self.action_mean[None]
        raw_action = {
            "world_vector": np.array(raw_actions[:, :3]),
            "rotation_delta": np.array(raw_actions[:, 3:6]),
            # range [0, 1]; 1 = open; 0 = close
            "open_gripper": np.array(raw_actions[:, 6:7]),
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )

        axangles = []
        for rotation_delta in action_rotation_delta:
            roll, pitch, yaw = rotation_delta
            ax, angle = euler2axangle(roll, pitch, yaw)
            axangle = ax * angle
            axangles.append(axangle[None])

        action["rot_axangle"] = np.concatenate(axangles, axis=0) * self.action_scale

        if self.policy_setup == "google_robot":
            print("google robot")
            current_gripper_action = raw_action["open_gripper"]

            # wow. what a mess when vectorized
            # alternative implementation
            relative_gripper_action = np.where(
                np.isnan(self.previous_gripper_action),
                np.zeros_like(self.previous_gripper_action),
                self.previous_gripper_action - current_gripper_action,
            )
            self.previous_gripper_action = current_gripper_action

            to_stick = np.logical_and(
                np.abs(relative_gripper_action) > 0.5,
                (self.sticky_action_is_on is False),
            )
            self.sticky_action_is_on = np.where(
                to_stick, True, self.sticky_action_is_on
            )
            self.sticky_gripper_action = np.where(
                to_stick, relative_gripper_action, self.sticky_gripper_action
            )

            self.gripper_action_repeat += self.sticky_action_is_on.astype(int)
            relative_gripper_action = np.where(
                self.sticky_action_is_on,
                self.sticky_gripper_action,
                relative_gripper_action,
            )

            reset = self.gripper_action_repeat == self.sticky_gripper_num_repeat
            self.sticky_action_is_on = np.where(reset, False, self.sticky_action_is_on)
            self.gripper_action_repeat = np.where(reset, 0, self.gripper_action_repeat)
            self.sticky_gripper_action = np.where(
                reset, 0.0, self.sticky_gripper_action
            )

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            # binarize gripper action to 1 (open) and -1 (close)
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0] * self.batch_size)

        return raw_action, action


class ResizeImageWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        resize_size: Optional[Union[Tuple, Sequence[Tuple]]],
    ):
        super().__init__(env)
        assert isinstance(
            self.observation_space, gym.spaces.Dict
        ), "Only Dict observation spaces are supported."
        spaces = self.observation_space.spaces
        self.resize_size = resize_size

        logging.info(f"Resizing images:")
        for k, v in self.observation_space.sample().items():
            if isimg(v):
                spaces[k] = gym.spaces.Box(
                    low=0,
                    high=255,  # pixel brightness
                    shape=resize_size + (3,),
                    dtype=np.uint8,
                )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        for k, v in observation.items():
            if isimg(v):
                image = tf.image.resize(
                    v, size=self.resize_size, method="lanczos3", antialias=True
                )
                image = tf.cast(
                    tf.clip_by_value(tf.round(image), 0, 255), tf.uint8
                ).numpy()
                observation[k] = image
        return observation


def mk_envs(n_envs=cfg.inference_size):
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

    import gymnasium as gym
    import improve.wrapper as W
    from improve.fm.batch_octo import BatchedOctoInference
    from octo.utils import gym_wrappers as GW
    from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                                  VecMonitor, VecVideoRecorder)

    def _init() -> gym.Env:

        env = simpler.make(
            cfg.task,
            # cant find simpler-img if you specify the mode
            render_mode="cameras",
            # max_episode_steps=max_episode_steps,
            renderer_kwargs={
                "offscreen_only": True,
                "device": "cuda:0",
            },
            # **extra,
        )

        # oxes does this
        # env = W.ActionRescaleWrapper(env)
        # env = W.StickyGripperWrapper(env, task=cfg.task)

        env = W.ExtraObservationWrapper(env, use_image=True)
        env = W.FlattenKeysWrapper(env)
        env = W.FilterKeysWrapper(env, keys=["simpler-img"])
        env = W.SuccessInfoWrapper(env)
        # print(env.observation_space)

        # TODO octo needs gymnasium as gym

        # requires key.startswith('image_')
        env = ResizeImageWrapper(env, (256, 256))
        # env = GW.HistoryWrapper(env, 2)
        # env = GW.TemporalEnsembleWrapper(env, 4)

        # env = GW.RHCWrapper(env, 2)
        return env

    # batch must match n_envs :(
    venv = SubprocVecEnv([_init for _ in range(n_envs)])
    # venv = VecMonitor(venv)  # attach this so SB3 can log reward metrics

    venv.seed(0)
    venv.reset()
    venv = W.VecRecord(venv, osp.join("log_dir", "train"), use_wandb=True)
    return venv


class EvalCallback:

    def __init__(self, venv, step_func, transform=None, oxes=None):
        self.venv = venv
        self.step_func = step_func
        self.transform = transform
        self.oxes = oxes
        self.rescaler = ActionRescaler()

    def __call__(self, i=None):

        if self.oxes is not None:
            self.oxes.reset()

        dones = np.zeros(cfg.inference_size, dtype="int")

        rewards = np.zeros(cfg.inference_size)
        lengths = np.zeros(cfg.inference_size, dtype="int")
        obs = self.venv.reset()  # venv reset has no info

        obs = self.transform(obs) if self.transform is not None else obs
        obs = self.oxes.pre_step(obs) if self.oxes is not None else obs

        states = None
        episode_starts = np.ones((self.venv.num_envs,), dtype=bool)

        # print(jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), obs))

        bar = tqdm("eval callback", total=120, leave=False)
        with bar:
            while not dones.all():

                # actions = np.array(self.step_func(obs))
                actions = self.step_func(obs)
                if self.oxes is not None:
                    raw, actions = self.oxes(actions)

                actions = self.rescaler.dict2act(actions)

                # names = ["x", "y", "z", "yaw", "pitch", "roll", "gripper"]
                # for i, n in enumerate(names):
                # wandb.log({f"pred/{n}": wandb.Histogram(actions[:, i])})

                obs, rew, done, info = self.venv.step(actions)
                obs = self.transform(obs) if self.transform is not None else obs
                obs = self.oxes.pre_step(obs) if self.oxes is not None else obs
                # done = terminated or truncated

                rewards = rewards + (rew * np.logical_not(rewards))
                lengths = lengths + (1 * np.logical_not(rewards))
                dones = np.logical_or(dones, done)

                bar.set_description(f"rewards: {rewards.sum()}")
                bar.update()

        stats = {"SR": rewards.mean(), "length": lengths.mean()}
        return stats


def step_randoms(*args):
    import numpy as np

    return np.random.rand(cfg.inference_size, 4, 7)


def main():

    print("Using wandb")
    wrun = wandb.init(
        project="lora",
        dir=osp.join(osp.expanduser("~"), "improve_logs"),  # cfg.callback.log_path,
        job_type="train",
        # sync_tensorboard=True,
        monitor_gym=True,
        config=asdict(cfg),  # OC.to_container(cfg, resolve=True),
        # name="debug"
    )
    wandb.config.update({"name": wrun.name})

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

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

    # pretrained = pretrained_model
    # dataset_id = "bridge_dataset"
    # action_mean = pretrained.dataset_statistics[dataset_id]["action"]["mean"]
    # action_std = pretrained.dataset_statistics[dataset_id]["action"]["std"]

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

    dataset = map(
        shard,
        iter(lora_octo.octo_dataset(cfg.batch_size)),
    )
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
    """
    # config["model"]["heads"]["value"] = ModuleSpec.create(
    #     MSEActionHead,
    #     pred_horizon=4,  # 50 for aloha but thats too much for simpler
    #     max_action=1.0,
    #     action_dim=1,
    #     readout_key="readout_value",
    # )

    ### TODO: add the critic head
    config["model"]["heads"]["value"] = ModuleSpec.create(
        MSECriticHead,
        predictions=1,
        obs_horizon=2,
        pred_horizon=4,
        chunk_size=5,
        max_critic=1.0,
        readout_key="readout_value",
    )

    config["model"]["readouts"]["value"] = 2
    print(config)

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

        if ("heads_action" in path and "diffusion_model" in path) and (
            "Dense_0" in path or "Dense_1" in path
        ):
            print(f"Using LoRA with dim={dim} for param {joined}")
            return dim

        if "heads_value" in path:
            return lorax.LORA_FULL

        print(joined)
        return lorax.LORA_FREEZE

        if False:
            print(f"Fully finetuning param {path}")
            return lorax.LORA_FULL

    lora_spec = lorax.simple_spec(
        model.params, decision_fn=decision_fn, tune_vectors=True
    )
    # print(lora_spec)

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

    lrschedule = optax.cosine_decay_schedule(3e-4, cfg.train_steps)
    # learning_rate = optax.join_schedules( [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100])
    tx = optax.adamw(learning_rate=lrschedule, weight_decay=1e-4, mu_dtype=jnp.bfloat16)
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
    ### TODO: don't need to do this twice??
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

        value_loss, value_metrics = bound_module.heads["value"].loss(
            transformer_embeddings,
            batch["action"],  # NEW: Q(s,a)
            batch["value"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train,
        )

        loss = action_loss + 0.5 * value_loss
        metrics = {"action": action_metrics, "value": value_metrics}
        return loss, metrics

    loss_fn = mk_octo_adv_loss(model, beta=3.0)
    print("using octo AWAC loss")

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        # donate_argnums=0,
    )
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    @lorax.lora
    def _model_step(params, batch, rng, train=False):
        """for evaluation in env"""
        # use the params and rng from the state
        bound_module = model.module.bind(
            {"params": params}, rngs={"dropout": train_state.rng}
        )

        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )

        actions = bound_module.heads["action"].predict_action(
            transformer_embeddings,
            rng=train_state.rng,
            train=False,
        )

        # print(actions)
        # print(actions.shape)
        return actions

    # selects best of 5*20 proposals
    # _model_step = mk_model_step(model, train_state)

    #
    #
    #

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(dp_sharding),
        out_shardings=(replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        # donate_argnums=0,
    )
    def model_step(batch):
        actions = _model_step(train_state.params, batch, train_state.rng)
        return actions[: cfg.inference_size]

    def transform(obs):
        zeros = jax.tree.map(
            lambda arr: jnp.zeros(
                (cfg.batch_size - cfg.inference_size, *arr.shape[1:])
            ),
            obs,
        )
        # zeros = jax.tree.map(lambda arr: jnp.zeros(arr), gapspec)
        obs = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), obs, zeros)
        obs["task"] = {"language_instruction": lang}
        return obs

    venv = mk_envs(cfg.inference_size)
    from improve.fm.oxes import OXESimplerInference, PolicyStepper

    stepper = PolicyStepper(
        model_type="func",
        dataset_id="bridge_dataset",
        func=model_step,
        transform=transform,
    )

    oxes = OXESimplerInference(stepper, batch_size=cfg.inference_size)
    oxes.reset(descs)

    def og_step(obs):
        raw, act = oxes.step(obs)
        return act

    evalcallback = EvalCallback(venv, og_step)

    #
    #
    #

    timer = Timer()

    # run finetuning loop
    logging.info("Starting finetuning...")
    for i in tqdm(range(cfg.train_steps), total=cfg.train_steps, dynamic_ncols=True):

        # timer.tick("total")

        with timer("dataset"):
            batch = next(dataset)
            batch["task"] = {"language_instruction": lang}
            # batch = example_batch
            # batch = process_batch(batch, text_processor)

        with timer("train"):
            train_state, train_info = train_step(train_state, batch)

        if (i + 1) % 100 == 0:
            # if (i + 1) % 100 == 0:
            train_info = jax.device_get(train_info)

            lr = lrschedule(i)
            # print(f"Step {i}: LR: {lr}")
            train_info.update({"learning_rate": lr})

            evals = {}
            # if (i+1) % 500 == 0:
            if (i + 1) % 100 == 0:
                with timer("rollout"):
                    evals = evalcallback(i)
                    evals = {"eval": evals}

            info = {
                "training": train_info,
                "timer": timer.get_average_times(),
                **evals,
            }
            print(info)
            with timer("wandb"):
                wandb.log(du.flatten(info, delim="/"), step=i)

        # if (i + 1) % 1000 == 0:
        # save checkpoint
        # train_state.model.save_pretrained(step=i, checkpoint_path=cfg.save_dir)

        # timer.tock("total")


if __name__ == "__main__":
    main()
