import itertools
import json
import os
import os.path as osp
import random
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from improve import cn
from improve.env.action_rescale import ActionRescaler
from improve.wrapper import dict_util as du

# decord comes last
_ = None
import decord


def decord2mp4(p):

    # Use decord to read the video
    # decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(p, ctx=decord.cpu(0))

    # Convert the video frames to a NumPy array
    frames = vr.get_batch(range(len(vr)))
    # frames = frames.numpy()
    frames = frames.asnumpy()
    return frames


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
            return torch.stack(lst) if not lst[0] is None else None
            return jnp.stack(lst) if not lst[0] is None else None

    return list2dict(samples)


LowDimKeys = [
    "agent_qpos-sin",
    "agent_qpos-cos",
    "agent_qvel",
    "eef-pose",
]

RPKeys = ["agent_partial-action"]
OracleKeys = ["obj-pose", "obj-wrt-eef"]    ### CHANGED (added obj pose to OracleKeys)
ImageKeys = ["simpler-img"]


class DefaultTransform:
    def __init__(self):
        self.scaler = ActionRescaler(cn.Strategy.CLIP, residual_scale=1.0)

    def filter_keys(self, obs):

        obs = {k: v for k, v in obs.items() if k in LowDimKeys + OracleKeys}
        # obs = {k: v for k, v in obs.items() if k in ImageKeys}

        if "simpler-img" in obs:
            dims = (0,3,1,2) if len(obs['simpler-img'].shape) == 4 else (2,0,1)
            obs["simpler-img"] = np.transpose(obs["simpler-img"], dims)
        return obs

    def __call__(self, x):

        actions = np.array([x["actions"]])
        # actions = self.scaler.scale_action(np.array(x["actions"]))
        if "agent_partial-action" in x["obs"]:
            fm = self.scaler.scale_action(np.array(x["obs"]["agent_partial-action"]))
            # actions[:,:-1] += fm[:,:-1]
            actions[:, -1] = fm[:, -1]
        actions = self.scaler.unscale_for_obs(actions)[0]

        x["obs"] = self.filter_keys(x["obs"])
        x["next_obs"] = self.filter_keys(x["next_obs"])
        x['actions'] = actions
        return x

def steps2batch(dataset):

    def _mk_batch(steps):
        obs = du.stack([s[0] for s in steps], force=True)
        next_obs = du.stack([s[1] for s in steps], force=True)
        actions = np.array([s[2] for s in steps])
        rewards = np.array([s[3] for s in steps])
        dones = np.array([s[4] for s in steps])
        infos = [s[5] for s in steps]  # i guess these arent stacked?
        return obs, next_obs, actions, rewards, dones, infos

    queue = []
    for sample in dataset:
        steps = list(ep2step(sample))
        for s in steps:
            queue.append(s)
            if len(queue) == 8:
                batch = _mk_batch(queue)
                queue = []
                yield batch



class MyOfflineDS(IterableDataset):
    def __init__(self, root=".", seq=2, transform=None, shift_reward=False):
        super(MyOfflineDS).__init__()

        self.root = root
        
        # TODO: comment this out
        self.samples = list({x.split(".")[0] for x in os.listdir(root) if x.endswith("pt")})
        self.samples = [x for x in self.samples if not (x.endswith("py"))]

        self.transform = transform
        self.seq = seq
        
        self.shift_reward = shift_reward

    def decode(self, idx):
        # print(idx)
        path = osp.join(self.root, f"{idx}.pt")
        data = torch.load(path)
        
        mp4_names = ["obs", "next_obs", "video"]
        
        for name in mp4_names:
            decoded_mp4 = decord2mp4(osp.join(self.root, f"{idx}.{name}.mp4")) 
            if name in data:
                data[name]['simpler-img'] = decoded_mp4
            else:
                data[name] = decoded_mp4
                
        # check for older datasets
        if "__key__" in data:
            del data["__key__"]
            
        ### CHANGED (added reward shifting)
        if self.shift_reward:
            data["rewards"] = data["rewards"] - 1.0
        
        # Add monte carlo reward
        for i in range(len(data["rewards"]) -2, -1, -1):
            data["rewards"][i] = 0.99 * data["rewards"][i+1]

        # TODO fix so that pt is written correctly
        infos = data["infos"].item()
        infos = du.flatten(infos, delim="/")
        infos = {k: np.array(v) for k, v in infos.items()}
        data["infos"] = du.nest(infos, delim="/")

        length = len(data["rewards"])
        for i in range(length // self.seq):
            if self.seq > 1:
                out = du.apply(data, lambda x: x[i * self.seq : (i + 1) * self.seq])
            else:
                out = du.apply(data, lambda x: x[i])

            out = self.transform(out) if self.transform is not None else out
            yield out

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        # print(worker)
        
        samples = (
            self.samples
            if worker is None
            else [
                x
                for i, x in enumerate(self.samples)
                if i % worker.num_workers == worker.id
            ]
        )
        # no cycle since it populates the buffer
        # samples = itertools.cycle(samples)

        for s in samples:
            yield from self.decode(s)


class SimpleDataset(IterableDataset):
    def __iter__(self):
        for i in range(10000):
            yield torch.tensor([i])


def get_json(f):
    with open(f, "r") as f:
        return json.load(f)


def process(f):
    proc = {
        "mp4": lambda x: np.array(decord2mp4(x)),
        "pt": lambda x: np.array(torch.load(x)),
        "json": lambda x: np.array(get_json(x)),
        "npz": lambda x: np.load(x)["arr_0"],
    }

    ext = f.split(".")[-1]
    return proc[ext](f)


def tar2pt(dname):
    """convert all files in a directory to pt files"""

    files = os.listdir(dname)
    keys = set([x.split(".")[0] for x in files])
    files = {k: [x for x in files if x.startswith(f"{k}.")] for k in keys}

    for k, v in tqdm(files.items()):
        v = [x for x in v if not x.endswith("pt")]
        # print(v)
        noext = lambda x: ".".join(x.split(".")[:-1])
        data = du.nest({noext(x): process(osp.join(dname, x)) for x in v}, delim=".")[k]
        # pprint(du.apply(data, lambda x: type(x)))

        torch.save(data, osp.join(dname, f"{k}.pt"))


def main():

    # set random seed
    # np.random.seed(42)
    # torch.manual_seed(42)
    # random.seed(42)

    HOME = os.environ["HOME"]
    # dataset = ["sunny-eon-12", "sleek-shadow-99"]
    # dataset = ["sunny-eon-12"]
    # dataset = ["sleek-cosmos-131"]
    # dataset = ["still-resonance-165"]

    dname = osp.join(HOME, "improve_logs", "dltarball", "train")
    # quit()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    ds = MyOfflineDS(root=dname, seq=1, transform=DefaultTransform())

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=16,
        # num_workers=0,
        num_workers=12,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    print(loader)
    for b in tqdm(loader, total=int(8503)):
        # print(du.apply(b, lambda x: x.shape))
        # quit()
        pass
    quit()


if __name__ == "__main__":
    main()
