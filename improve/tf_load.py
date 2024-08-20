import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import io
import torch
from pprint import pprint
from improve.wrapper import dict_util as du
from improve import cn
from improve.env.action_rescale import ActionRescaler
import time

from improve.fm.cache import load_task_embedding

import jax
import jax.numpy as jnp

_ = None
import decord

ImageKeys = ["simpler-img"]
scaler = ActionRescaler(cn.Strategy.CLIP, residual_scale=1.0)

feature_description = {
    'sample_number': tf.io.FixedLenFeature([], tf.int64),
    'obs': tf.io.FixedLenFeature([], tf.string),
    'state': tf.io.FixedLenFeature([], tf.string),
    'video': tf.io.FixedLenFeature([], tf.string),
    'next_obs': tf.io.FixedLenFeature([], tf.string),
}

def decord2mp4(b):
    buffer = io.BytesIO(b)

    # Use decord to read the video
    # decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(buffer, ctx=decord.cpu(0))

    # Convert the video frames to a NumPy array
    frames = vr.get_batch(range(len(vr)))
    # frames = frames.numpy()
    frames = frames.asnumpy()
    return frames

def filter_keys(obs):
    obs = {k: v for k, v in obs.items() if k in ImageKeys}
    return obs

def preprocess(x: dict):

    proc = {
        "mp4": decord2mp4,
        "pt": lambda x: torch.load(io.BytesIO(x)),
    }

    x['state.pt'] = x['state']
    del x['state']

    x['obs.mp4'] = x['obs']
    del x['obs']

    x['video.mp4'] = x['video']
    del x['video']

    x['next_obs.mp4'] = x['next_obs']
    del x['next_obs']


    # checks file exten of key and applies the proc fucntion to the value, otherwise return as is
    def _process(k, v):
        return proc[k.split(".")[-1]](v) if k.split(".")[-1] in proc else v

    # applies _process func to every item in input dict x
    x = {k: _process(k, v) for k, v in x.items()}
    # removes file extensions from keys in dict
    x = {".".join(k.split(".")[:-1]): v for k, v in x.items()}

    x = du.nest(x, delim=".")

    # make sure infos exists
    if 'infos' not in x:
        x['infos'] = [None]

    # udpate obs and next_obs dicts inside state dict with obs and next_obs from x
    x['state']['obs'].update({'simpler-img': x['obs']})
    x['state']['next_obs'].update({'simpler-img': x['next_obs']})
    # sets infos in the state dict to infos from x
    x['state']['infos'] =  x['infos']
    # sets x to state dict in x. Keeps only the state dict but deletes everythign else
    x = x['state']
    # converst all pt tensors to numpy arrays
    x = du.apply(x, lambda x: x.numpy() if isinstance(x,torch.Tensor) else x)

    actions = np.array(x["actions"])

    # actions = scaler.scale_action(np.array(x["actions"]))
    if "agent_partial-action" in x["obs"]:
        fm = scaler.scale_action(np.array(x["obs"]["agent_partial-action"]))
        # actions[:,:-1] += fm[:,:-1]
        actions[:, -1] = fm[:, -1]
    actions = scaler.unscale_for_obs(actions)
    x['actions'] = actions

    # return x
    return (
        filter_keys(x["obs"]),
        filter_keys(x["next_obs"]),
        actions,
        x["rewards"],
        x["dones"],
        x["infos"],
    )

def tuple2dict(x):
    # breakpoint()
    return {
        "obs": jax.tree.map(lambda a: jnp.array(a), x[0]),
        "next_obs": jax.tree.map(lambda a: jnp.array(a), x[1]),
        "action": jnp.array(x[2]),
        "reward": jnp.array(x[3]),
        "done": jnp.array(x[4]),
    }

def future_actions(x: dict) -> None:
    actions = x["action"]
    n = actions.shape[0]

    # Create a new array of zeros with shape (n, 5, 7)
    new_actions = jnp.zeros((n, 5, 7))

    # Use roll to shift the array and vmap to apply it over a range of shifts
    def get_future_actions(idx):
        return jnp.stack(
            [jnp.roll(actions, -shift, axis=0)[idx] for shift in range(5)], axis=0
        )

    get_future_actions_vmap = jax.vmap(get_future_actions, in_axes=(0,))
    new_actions = get_future_actions_vmap(jnp.arange(n))

    # Handle edge cases where we roll beyond the length of the array
    mask = jnp.arange(n).reshape(-1, 1) + jnp.arange(5) < n
    # new_actions = jnp.where(mask[:, :, None], new_actions, actions[-1])

    # should predict 0s when done
    new_actions = jnp.where(mask[:, :, None], new_actions, jnp.zeros_like(actions[-1]))

    x["action"] = new_actions
    return x

def split_sample(x, n=8):
    l = len(x["done"])
    out = []
    for i in range(l):
        out.append(jax.tree.map(lambda a: a[i], x))
    return out

def parse_example(serialized_example):
    return tf.io.parse_single_example(serialized_example, feature_description)

def view_tfrecords(dataset, num_samples=1):
    for idx, sample in enumerate(dataset.take(num_samples)):
        print("idx:", idx)
        print(type(sample['obs']))
        print(f"Sample Number: {sample['sample_number'].numpy()}")
        print(f"Obs Video Size: {len(sample['obs'].numpy())} bytes")
        print(f"State Data Size: {len(sample['state'].numpy())} bytes")
        print(f"Main Video Size: {len(sample['video'].numpy())} bytes")
        print(f"Next Obs Video Size: {len(sample['next_obs'].numpy())} bytes")
        print("\n")

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
            if isinstance(lst[0], np.ndarray):
                return torch.tensor(np.stack(lst)) if not lst[0] is None else None
            if isinstance(lst[0], torch.Tensor):
                return torch.stack(lst) if not lst[0] is None else None
            if isinstance(lst[0], tf.Tensor):
                return tf.stack(lst) if not lst[0] is None else None

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

def is_batch(x):
    x = {k: v for k, v in x.items() if "__" not in k}
    sz = jax.tree.reduce(lambda s, arr: s and arr.shape[0] == 8, x, True) # FIXME: Hardcoded batch size
    return sz

def jax_to_numpy(data):
    """Recursively convert JAX arrays to numpy arrays."""
    if isinstance(data, jnp.ndarray):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: jax_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [jax_to_numpy(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(jax_to_numpy(v) for v in data)
    return data

def batch_and_collate(dataset, batch_size):
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) == batch_size:
            yield dict_collate(batch)
            batch = []

    if batch:
        yield dict_collate(batch)

def tensor2numpy(sample):
    entry = {
            'sample_number': sample['sample_number'].numpy(),
            'obs': sample['obs'].numpy(),
            'state': sample['state'].numpy(),
            'video': sample['video'].numpy(),
            'next_obs': sample['next_obs'].numpy()
        }
    return entry

def process_tfrecords(dataset):
    for sample in dataset:
        entry = tensor2numpy(sample)
        entry = preprocess(entry)
        entry = tuple2dict(entry)
        entry = future_actions(entry)
        entry = split_sample(entry)
        for sub_entry in entry:
            yield sub_entry

def process_dataset_2(dataset):
    for entry in dataset:
        entry = prepare_octo(entry)
        entry = get_mask(entry)
        entry = get_task(entry)
        entry = jax_to_numpy(entry)

        yield entry

def dataset_generator (tfrecord_files, batch_size=2, final_batch_size=8):
    def generator():
        # dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=4)
        dataset = dataset.map(parse_example, num_parallel_calls=4, deterministic=False)
        # dataset = dataset.map(parse_example)
        dataset = dataset.shard(num_shards=4, index=jax.process_index())
        dataset = dataset.shuffle(1024).prefetch(4)

        processed_samples = process_tfrecords(dataset)
        batched_samples = batch_and_collate(processed_samples, batch_size)
        processed_samples_2 = process_dataset_2(batched_samples)
        final_batches = batch_and_collate(processed_samples_2, final_batch_size)

        for x in final_batches:
            if is_batch(x):
                yield x

    return generator()

def mapped_dataset_generator (tfrecord_file, batch_size=2, final_batch_size=8):
    def generator():
        dataset = tf.data.TFRecordDataset(tfrecord_file)

        dataset = dataset.map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
        dataset = dataset.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(tensor2numpy, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(tuple2dict, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(future_actions, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(split_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        batched_samples = batch_and_collate(dataset, batch_size)

        processed_samples_2 = process_dataset_2(batched_samples)

        final_batches = batch_and_collate(processed_samples_2, final_batch_size)

        for x in final_batches:
            if is_batch(x):
                yield x

    return generator

def main():
    # tfrecord_file = '/home/ekuo/improve_logs/magic-universe-395/magic_universe.tfrecords'
    # dataset = dataset_generator(tfrecord_file)

    tfrecords_dir = '/home/ekuo/improve_logs/magic-universe-395/tfrecords'
    tfrecord_files = [os.path.join(tfrecords_dir, f) for f in os.listdir(tfrecords_dir) if f.endswith('.tfrecord')]
    dataset = dataset_generator(tfrecord_files)

    tfds.benchmark(dataset, batch_size=8, num_iter=128)
    tfds.benchmark(dataset, batch_size=8, num_iter=128)

    # example_batch = next(dataset)
    # print(type(dataset))
    # # example_batch = next(iter(dataset))
    # example_batch_spec = jax.tree.map(
    #     lambda arr: (arr.shape, str(arr.dtype)), example_batch
    # )
    # print('>'*20, "example_batch_spec")
    # pprint(example_batch_spec)
    # print(example_batch.keys())

if __name__ == "__main__":
    main()