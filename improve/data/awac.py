import io
import os
import os.path as osp
import random
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

import decord
import numpy as np
import torch
import torchvision.transforms as transforms
import webdataset as wds
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from improve import cn
from improve.env.action_rescale import ActionRescaler
from improve.wrapper import dict_util as du


def b2mp4():
    import imageio
    import numpy as np

    # Example dictionary with bytes as key
    # Assuming your dict is something like this
    # dict_with_bytes = {b'mp4_file': <mp4_bytes_data>}
    # Extract the bytes
    mp4_bytes = dict_with_bytes[b"mp4_file"]

    # Create a memory buffer from the bytes
    mp4_buffer = io.BytesIO(mp4_bytes)

    # Use imageio to read the video
    video_reader = imageio.get_reader(mp4_buffer, "ffmpeg")

    # Initialize a list to store frames
    frames = []

    # Read frames from the video
    for frame in video_reader:
        frames.append(frame)

    # Convert the list of frames to a numpy array
    video_array = np.array(frames)

    print(video_array.shape)


def decord2mp4(b):

    buffer = io.BytesIO(b)

    # Use decord to read the video
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(buffer, ctx=decord.cpu(0))

    # Convert the video frames to a NumPy array
    frames = vr.get_batch(range(len(vr)))
    frames = frames.numpy()
    # frames = frames.asnumpy()
    return frames


def find_tarballs(paths: List[str]):
    """walk the directory and find all tarballs"""
    paths = [paths] if isinstance(paths, str) else paths

    for p in paths:
        for root, dirs, files in os.walk(p):
            for file in files:
                if file.endswith(".tar"):
                    yield os.path.join(root, file)


LowDimKeys = [
    "agent_qpos-sin",
    "agent_qpos-cos",
    "agent_qvel",
    "eef-pose",
]

SourceTargetKeys = [
    "src-pose",
    "tgt-pose",
    "src-wrt-eef",
    "tgt-wrt-eef",
]

DrawerKeys = [
    "drawer-pose",
    "drawer-pose-wrt-eef",
]

RPKeys = ["agent_partial-action"]
OracleKeys = ["obj-wrt-eef"]
ImageKeys = ["simpler-img"]


def filter_keys(obs, task=None):

    if "simpler-img" in obs:
        obs["simpler-img"] = np.transpose(obs["simpler-img"], (0, 3, 1, 2))
    
    obs_keys = LowDimKeys
    if any(word in task for word in ["spoon", "near", "carrot", "cube"]):
        obs_keys += SourceTargetKeys
    elif "drawer" in task:
        obs_keys += DrawerKeys
    else:
        obs_keys += OracleKeys
    
    obs = {k: v for k, v in obs.items() if k in obs_keys} ### CHANGED
    # obs = {k: v for k, v in obs.items() if k in LowDimKeys + OracleKeys}#OracleKeys
    # obs = {k: v for k, v in obs.items() if k in LowDimKeys + SourceTargetKeys}#OracleKeys}  ### CHANGED
    # obs = {k: v for k, v in obs.items() if k in ImageKeys}
    return obs


scaler = ActionRescaler(cn.Strategy.CLIP, residual_scale=1.0)


def unscale(action):
    return scaler.unscale_for_obs(action)


def preprocess(x: dict, task=None):
    proc = {
        "mp4": decord2mp4,
        "npz": lambda x: x["arr_0"],
    }

    def _process(k, v):
        return proc[k.split(".")[-1]](v) if k.split(".")[-1] in proc else v

    x = {k: _process(k, v) for k, v in x.items()}
    x = {".".join(k.split(".")[:-1]): v for k, v in x.items()}
    x = du.nest(x, delim=".")

    actions = np.array(x["actions"])
    # actions = scaler.scale_action(np.array(x["actions"]))
    if "agent_partial-action" in x["obs"]:
        fm = scaler.scale_action(np.array(x["obs"]["agent_partial-action"]))
        # actions[:,:-1] += fm[:,:-1]
        actions[:, -1] = fm[:, -1]
    actions = scaler.unscale_for_obs(actions)

    return (
        filter_keys(x["obs"], task=task),  # CHANGE THIS
        filter_keys(x["next_obs"], task=task),
        actions,
        x["rewards"],
        x["dones"],
        x["infos"],
    )


def ep2step(ep: Tuple[Any]):

    epl = max([len(x) for x in ep])

    # for all e in ep if e is dict, return the i-th item of each k in dict
    # if e is a list, return the i-th item of each e
    # if e is a np.array, return the i-th item of each e
    # yield the i-th item of each e together at the same time

    def select(e, i):
        if isinstance(e, dict):
            return du.apply_mappable(e, lambda x: x[i])
        elif isinstance(e, list):
            return e[i]
        elif isinstance(e, np.ndarray):
            return e[i]

    for i in range(epl):
        yield tuple(select(e, i) for e in ep)


def mk_dataset(fnames, task=None):

    dataset = wds.DataPipeline(
        wds.SimpleShardList(fnames),
        # use resampled shards if you want to loop the dataset
        # wds.ResampledShards(fnames, deterministic=True),
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
        wds.decode(),
        # wds.to_tuple("png", "json"),
        wds.map(lambda x: preprocess(x, task=task)),
        # wds.map(ep2step),
        # For IterableDataset objects, the batching needs to happen in the dataset.
        # wds.batched(16),
    )

    loader = wds.WebLoader(
        dataset,
        # batch_size=None,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,  # Persistent workers
    )

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    # loader = loader.unbatched().shuffle(1000).batched(16)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    # loader = loader.with_epoch(1282 * 100 // 64)

    return dataset


def main():

    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    HOME = os.environ["HOME"]
    dataset = ["sunny-eon-12", "sleek-shadow-99"]
    dataset = ["sunny-eon-12"]
    dataset = ["sleek-cosmos-131"]
    exp_root = [osp.join(HOME, "improve_logs", x) for x in dataset]

    # dnames = [osp.join(e, "eval") for e in exp_root]
    dnames = [osp.join(e, "train") for e in exp_root]
    fnames = list(find_tarballs(dnames))
    dataset = mk_dataset(fnames)

    reward = 0
    grippers = {}
    for batch in tqdm(dataset):
        steps = list(ep2step(batch))
        # print(steps[0])
        for s in steps:
            # print(reward)
            # print(s[1].keys())
            # print(s[2][-1])
            reward += s[3]

            print(s[0].keys())

            g = round(s[2][-1], 2)
            grippers[g] = grippers.get(g, 0) + 1

    print(grippers)

    # print([type(b) for b in steps[0]])
    # pass
    # quit()
    # pprint({k: type(v) for k, v in batch.items()})

    # print(batch.keys())
    # print(batch["next_obs.simpler-img.mp4"].shape)

    # print(batch['actions.json'])
    # print(len(batch['actions.json']))
    # print(image.size, json)


if __name__ == "__main__":
    main()
