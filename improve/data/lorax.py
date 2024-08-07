import io
import json
import os
import os.path as osp
import random
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from improve import cn
from improve.env.action_rescale import ActionRescaler
from improve.wrapper import dict_util as du

# decord last
_ = None
import decord


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


def b2npz(b):
    x = np.load(io.BytesIO(b), encoding="bytes")["arr_0"].astype(np.float32)
    return torch.Tensor(x)


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

RPKeys = ["agent_partial-action"]
OracleKeys = ["obj-wrt-eef"]
ImageKeys = ["simpler-img"]


def filter_keys(obs):

    # if "simpler-img" in obs:
    # obs["simpler-img"] = np.transpose(obs["simpler-img"], (0, 3, 1, 2))

    # obs = {k: v for k, v in obs.items() if k in LowDimKeys + OracleKeys}
    obs = {k: v for k, v in obs.items() if k in ImageKeys}
    return obs


# scaler = ActionRescaler(cn.Strategy.CLIP, residual_scale=1.0)


def unscale(action):
    """ this is un-scaling for Octo with widowX robot """
    print('unscale called')
    m = [0.00021161, 0.00012614, -0.00017022, -0.00015062, -0.00023831, 0.00025646, 0.0]
    mean = np.array(m)
    s = [0.00963721, 0.0135066, 0.01251861, 0.02806791, 0.03016905, 0.07632624, 1.0]
    std = np.array(s)

    out = (action - mean[None]) / std[None]
    return out


def preprocess(x: dict):
    proc = {
        "mp4": decord2mp4,
        "pt": lambda x: torch.load(io.BytesIO(x)),
    }

    def _process(k, v):
        return proc[k.split(".")[-1]](v) if k.split(".")[-1] in proc else v

    x = {k: _process(k, v) for k, v in x.items()}
    x = {".".join(k.split(".")[:-1]): v for k, v in x.items()}
    x = du.nest(x, delim=".")

    if "infos" not in x:
        x["infos"] = [None]

    x["state"]["obs"].update(**x["obs"])
    x["state"]["next_obs"].update(**x["next_obs"])
    x["state"]["infos"] = x["infos"]
    x = x["state"]
    x = du.apply(x, lambda x: x.numpy() if isinstance(x, torch.Tensor) else x)

    actions = np.array(x["actions"])

    # actions = scaler.scale_action(np.array(x["actions"]))
    if "agent_partial-action" in x["obs"]:
        # dont add partial action since main action already has
        # actions[:,:-1] += fm[:,:-1]
        # fm = scaler.scale_action(np.array(x["obs"]["agent_partial-action"]))
        fm = np.array(x["obs"]["agent_partial-action"])
        actions[:, -1] = fm[:, -1]

    actions = unscale(actions)
    x["actions"] = actions

    # return x
    return (
        filter_keys(x["obs"]),
        filter_keys(x["next_obs"]),
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

    return [tuple(select(e, i) for e in ep) for i in range(epl)]


def mk_dataset(fnames):

    dataset = wds.DataPipeline(
        # wds.SimpleShardList( fnames),
        # use resampled shards if you want to loop the dataset
        wds.ResampledShards(fnames, deterministic=True),
        # at this point we have an iterator over all the shards
        # wds.shuffle(100), # shuffles the shards
        #
        wds.split_by_worker,
        # add wds.split_by_node here if you are using multiple nodes
        # wds.split_by_node,
        #
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(),
        # this shuffles the samples in memory
        wds.shuffle(500),  # shuffles the samples... too much shuffle will kill
        # this decodes the images and json
        wds.decode(),
        # wds.to_tuple("png", "json"),
        wds.map(preprocess),
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


def increase_file_lim():
    import resource

    # Set soft limit to 4096 and hard limit to 8192
    soft, hard = 4096, 8192
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    # Verify the changes
    print("Soft limit:", resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    print("Hard limit:", resource.getrlimit(resource.RLIMIT_NOFILE)[1])


def main():

    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    HOME = os.environ["HOME"]
    dataset = ["sunny-eon-12"]
    exp_root = [osp.join(HOME, "improve_logs", x) for x in dataset]

    # dnames = [osp.join(e, "eval") for e in exp_root]
    dnames = [osp.join(e, "train") for e in exp_root]
    fnames = list(find_tarballs(dnames))

    os.system("ulimit -n 4096")
    # increase_file_lim()
    print(fnames)
    dataset = mk_dataset(fnames)

    loader = wds.WebLoader(
        dataset,
        # batch_size=None,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,  # Persistent workers
    )
    # loader = DataLoader( dataset, batch_size=16, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True,)

    for b in tqdm(loader):
        pass
    quit()

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
