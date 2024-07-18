import copy
import functools
import io
import os.path as osp
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import imageio
import numpy as np
import webdataset as wds
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv, VecEnvObs,
                                                           VecEnvStepReturn,
                                                           VecEnvWrapper)

import wandb
from improve.wrapper import dict_util as du


def np2mp4b(data):
    vbytes = io.BytesIO()
    writer = imageio.get_writer(vbytes, format="mp4", mode="I", fps=5)

    for frame in data:
        writer.append_data(frame)
    writer.close()
    vbytes.seek(0)  # Rewind the file-like object to the beginning
    return vbytes.getvalue()


def np2npzb(data):
    buf = io.BytesIO()
    np.savez_compressed(buf, data)
    buf.seek(0)
    return buf.getvalue()


def isimg(o):
    if isinstance(o, np.ndarray) and o.ndim == 3:
        return o.shape[-1] in [1, 3, 4]
    return False


def try_ex(func, *args, **kwargs):
    """could be a decorator one day"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return None


class VecRecord(VecEnvWrapper):

    def __init__(self, venv: VecEnv, output_dir, use_wandb=True):

        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface

        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_count = 0
        self.t_start = time.time()
        self.use_wandb = use_wandb

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        # things for the shard writer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        now = time.strftime("%Y%m%d")
        self.fname = osp.join(self.output_dir, f"{now}-%06d.tar")
        self.shard = wds.ShardWriter(self.fname)
        # sink.write(sample)

        self.episodes = [[] for _ in range(self.num_envs)]
        self.actions = None
        self.renders = None
        self.last_obs = None

        self._elapsed_steps = 0
        self._episode_id = list(range(self.num_envs))
        self.id_counter = max(self._episode_id)

        self._episode_data = []
        self._episode_info = {}

    def reset(self, **kwargs):

        # if self.episodes:
        # self.flush_trajectory(ignore_empty_transition=True)
        # self.flush_video(ignore_empty_transition=True)

        obs = self.venv.reset(**kwargs)
        obs = du.todict(obs)
        reset_kwargs = copy.deepcopy(kwargs)

        self.last_obs = obs

        # Clear cache
        self._elapsed_steps = 0
        # self._episode_id += 1
        self._episode_data = []
        self._episode_info = {}
        self._render_images = []

        self.actions = None
        # state = self.env.get_state()
        self.data = dict(
            # s=state,
            o=copy.deepcopy(obs),
            a=None,
            r=None,
            terminated=None,
            truncated=None,
            info=None,
        )

        """
        self._episode_data.append(data)
        self._episode_info.update(
            episode_id=self._episode_id,
            episode_seed=getattr(self.unwrapped, "_episode_seed", None),
            reset_kwargs=reset_kwargs,
            control_mode=getattr(self.unwrapped, "control_mode", None),
            elapsed_steps=0,
        )
        """

        info = None
        from pprint import pprint

        pprint(obs.keys())
        return obs

    def step_async(self, actions: np.ndarray):
        self.actions = actions
        # episodes[-1].append(actions)
        self.venv.step_async(actions)

    def process_obs(self, obs):
        pass
        # undo this
        # return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def step_wait(self) -> VecEnvStepReturn:

        obs, rewards, dones, infos = self.venv.step_wait()
        # obs = self.process_obs(obs)

        transitions = {
            "obs": self.last_obs,
            "next_obs": obs,
            "actions": self.actions,
            "rewards": rewards,
            "dones": dones,
            # "infos": infos,
        }

        self.last_obs = obs
        # print(du.apply(infos, lambda x: type(x) ))

        transitions = [
            du.apply(transitions, lambda x: x[i]) for i in range(self.num_envs)
        ]

        for i in range(self.num_envs):
            transitions[i]["info"] = infos[i]
            self.episodes[i].append(transitions[i])

        renders = self.env_method("render")
        renders = np.expand_dims(renders, axis=1)

        if self.renders is None:
            self.renders = [r for r in renders]
        else:
            for i, r in enumerate(renders):
                tmp = [x for x in [self.renders[i], r] if x is not None]
                self.renders[i] = np.concatenate(tmp, axis=0)

        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])

        for i in range(len(dones)):
            if dones[i]:
                self.flush(i)

                info = infos[i].copy()

                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t_start, 6),
                }

                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                new_infos[i] = info

        # _infos = du.concat(new_infos) if len(new_infos) > 1 else new_infos[0]

        # for k, v in du.flatten(_infos, delim="/").items():
        # k = osp.join("stats", "info", k)

        return obs, rewards, dones, new_infos

    def flush(self, i):

        print(f"flushing {i}")

        ep = self.episodes[i]

        obs = [x["obs"] for x in ep]
        obs = du.stack(obs)

        # TODO fix this... youre essentially writing 2x the data
        next_obs = [x["next_obs"] for x in ep]
        next_obs = du.stack(next_obs)

        # print(du.apply(obs, lambda x: type(x)))
        # print(obs[0]['agent_base_pose'])

        frames = self.renders[i][:, :, 512:1024]
        vid = np2mp4b(frames)
        self.renders[i] = None

        # print(len(ep))
        # print(ep[0].keys())

        rewards = functools.reduce(lambda x, y: x + [float(y["rewards"])], ep, [])
        dones = functools.reduce(lambda x, y: x + [float(y["dones"])], ep, [])
        actions = functools.reduce(lambda x, y: x + [y["actions"].tolist()], ep, [])

        infos = [du.todict(x["info"]) for x in ep]
        infos = du.concat(infos)
        infos = du.apply(infos, lambda x: x.tolist())

        id = self._episode_id[i]
        self.id_counter += 1
        self._episode_id[i] = self.id_counter

        obs = {
            f'obs.{k}.{"mp4" if isimg(v[0]) else "npz"}': (
                np2mp4b(v) if isimg(v[0]) else np2npzb(v)
            )
            for k, v in obs.items()
        }

        next_obs = {
            f'next_obs.{k}.{"mp4" if isimg(v[0]) else "npz"}': (
                np2mp4b(v) if isimg(v[0]) else np2npzb(v)
            )
            for k, v in next_obs.items()
        }

        sample = {
            "__key__": f"{id}",
            **obs,
            **next_obs,
            "rewards.json": rewards,
            "actions.json": actions,
            "dones.json": dones,
            "video.mp4": vid,
            "infos.json": infos,
        }
        self.shard.write(sample)

        if self.use_wandb:
            caption = f"ep_id={id} | reward={sum(rewards)} | {'success' if sum(rewards) > 0 else 'failure'}"
            videos = {
                f"videos/obs.vid": wandb.Video(
                    frames.transpose(0, 3, 1, 2),
                    fps=5,
                    caption=caption,
                )
            }
            wandb.log(videos)

        self.episodes[i] = []

    def close(self) -> None:
        for i in range(self.num_envs):
            try_ex(self.flush(i))
        self.shard.close()

        return super().close()


class TMP(gym.Wrapper):
    """from maniskill2 ... used for hdf5"""

    def __init__(
        self,
        env,
        output_dir,
        trajectory_name=None,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        clean_on_close=True,
    ):
        super().__init__(env)

        # Use a separate json to store non-array data
        self._json_path = self._h5_file.filename.replace(".h5", ".json")
        self._json_data = dict(
            env_info=parse_env_info(self.env),
            commit_info=get_commit_info(),
            episodes=[],
        )

        self.save_video = save_video
        self.info_on_video = info_on_video
        self._render_images = []

        # Avoid circular import
        from mani_skill2.envs.mpm.base_env import MPMBaseEnv

        if isinstance(env.unwrapped, MPMBaseEnv):
            self.init_state_only = True
        else:
            self.init_state_only = False

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1

        if self.save_trajectory:
            state = self.env.get_state()
            data = dict(
                s=state,
                o=copy.deepcopy(obs),
                a=action,
                r=rew,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            self._episode_data.append(data)
            self._episode_info["elapsed_steps"] += 1
            self._episode_info["info"] = info

        if self.save_video:
            image = self.env.render()

            if self.info_on_video:
                scalar_info = extract_scalars_from_info(info)
                extra_texts = [
                    f"reward: {rew:.3f}",
                    "action: {}".format(",".join([f"{x:.2f}" for x in action])),
                ]
                image = put_info_on_image(image, scalar_info, extras=extra_texts)

            self._render_images.append(image)

        return obs, rew, terminated, truncated, info

    def flush_trajectory(self):

        sample = {"__key__": f"{self._episode_id}"}

        traj_id = "traj_{}".format(self._episode_id)
        group = self._h5_file.create_group(traj_id, track_order=True)

        # Observations need special processing
        obs = [x["o"] for x in self._episode_data]

        if isinstance(obs[0], dict):

            obs = [flatten_dict_keys(x) for x in obs]
            obs = {k: [x[k] for x in obs] for k in obs[0].keys()}
            obs = {k: np.stack(v) for k, v in obs.items()}

            for k, v in obs.items():

                subgroups = k.split("/")[:-1]

                if "rgb" in k and v.ndim == 4:
                    sample[f"obs.{k}.gzip"] = v
                elif "depth" in k and v.ndim in (3, 4):
                    # NOTE(jigu): uint16 is more efficient to store at cost of precision
                    if not np.all(np.logical_and(v >= 0, v < 2**6)):
                        raise RuntimeError(
                            "The depth map({}) is invalid with min({}) and max({}).".format(
                                k, v.min(), v.max()
                            )
                        )
                    v = (v * (2**10)).astype(np.uint16)
                    sample[f"obs.{k}.gzip"] = v

                elif "seg" in k and v.ndim in (3, 4):
                    assert (
                        np.issubdtype(v.dtype, np.integer) or v.dtype == np.bool_
                    ), v.dtype
                    sample[f"obs.{k}.gzip"] = v
                else:
                    sample[f"obs.{k}.gzip"] = v

        elif isinstance(obs[0], np.ndarray):
            obs = np.stack(obs)
            sample["obs.npz"] = obs
        else:
            print(obs[0])
            raise NotImplementedError(type(obs[0]))

        if len(self._episode_data) == 1:
            action_space = self.env.action_space
            assert isinstance(action_space, spaces.Box), action_space
            actions = np.empty(
                shape=(0,) + action_space.shape,
                dtype=action_space.dtype,
            )
            dones = np.empty(shape=(0,), dtype=bool)
        else:
            # NOTE(jigu): The format is designed to be compatible with ManiSkill-Learn (pyrl).
            # Record transitions (ignore the first padded values during reset)
            actions = np.stack([x["a"] for x in self._episode_data[1:]])
            # NOTE(jigu): "dones" need to stand for task success excluding time limit.
            dones = np.stack([x["info"]["success"] for x in self._episode_data[1:]])

        # Only support array like states now
        env_states = np.stack([x["s"] for x in self._episode_data])

        # Dump
        sample["actions.npz"] = actions.astype(np.float32)
        sample["dones.npz"] = dones.astype(bool)

        if self.init_state_only:
            group.create_dataset("env_init_state", data=env_states[0], dtype=np.float32)
        else:
            group.create_dataset("env_states", data=env_states, dtype=np.float32)

        sample["info.json"] = self._episode_info
        # Handle JSON

        if verbose:
            print("Record the {}-th episode".format(self._episode_id))

    def flush_video(self, suffix="", verbose=False, ignore_empty_transition=False):
        if not self.save_video or len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return

        video_name = "{}".format(self._episode_id)
        if suffix:
            video_name += "_" + suffix
        images_to_video(
            self._render_images,
            str(self.output_dir),
            video_name=video_name,
            fps=20,
            verbose=verbose,
        )


from omegaconf import OmegaConf as OC

import hydra
import improve
import improve.hydra.resolver


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    if cfg.job.wandb.use:
        print("Using wandb")
        run = wandb.init(
            project="residualrl-maniskill2demo",
            dir=cfg.callback.log_path,
            job_type="train",
            # sync_tensorboard=True,
            monitor_gym=True,
            name=cfg.job.wandb.name,
            group=cfg.job.wandb.group,
            config=OC.to_container(cfg, resolve=True),
        )
        wandb.config.update({"name": run.name})

    import simpler_env as simpler

    from improve.env import make_env
    from improve.wrapper.simpler.misc import FlattenKeysWrapper

    num_envs = 2
    env = SubprocVecEnv([make_env(cfg) for _ in range(num_envs)])

    env = VecRecord(env, output_dir=".", use_wandb=False)
    env.seed(cfg.job.seed)
    things = env.reset()

    print(type(things[0]), type(things[1]))
    quit()

    for _ in range(1000):
        actions = np.array([env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
