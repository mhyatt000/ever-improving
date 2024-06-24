"""
taken from maniskill2 to adapt for wandb
"""

import copy
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mani_skill2.utils.common import (extract_scalars_from_info,
                                      flatten_dict_keys)
from mani_skill2.utils.visualization.misc import (images_to_video,
                                                  put_info_on_image)


def parse_env_info(env: gym.Env):
    # spec can be None if not initialized from gymnasium.make
    env = env.unwrapped
    if env.spec is None:
        return None
    if hasattr(env.spec, "_kwargs"):
        # gym<=0.21
        env_kwargs = env.spec._kwargs
    else:
        # gym>=0.22
        env_kwargs = env.spec.kwargs
    return dict(
        env_id=env.spec.id,
        max_episode_steps=env.spec.max_episode_steps,
        env_kwargs=env_kwargs,
    )


class WandbVideoWrapper(gym.Wrapper):
    """Record trajectories or videos for episodes.
    The trajectories are stored in HDF5.

    Args:
        env: gym.Env
        output_dir: output directory
        save_video: whether to save video
        render_mode: rendering mode passed to `env.render`
        save_on_reset: whether to save the previous trajectory automatically when resetting.
            If True, the trajectory with empty transition will be ignored automatically.

        for mani_skill2.utils.wrappers import RecordEpisode only
        clean_on_close: whether to rename and prune trajectories when closed.
            See `clean_trajectories` for details.
    """

    def __init__(
        self,
        env,
        output_dir,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        clean_on_close=True,
    ):
        super().__init__(env)

        self.output_dir = Path(output_dir)
        if save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_on_reset = save_on_reset

        self._elapsed_steps = 0
        self._episode_id = -1
        self._episode_data = []
        self._episode_info = {}

        self.clean_on_close = clean_on_close

        self.save_video = save_video
        self.info_on_video = info_on_video
        self._render_images = []

    def reset(self, **kwargs):

        if self.save_on_reset and self._episode_id >= 0:
            if self._elapsed_steps == 0:
                self._episode_id -= 1
            self.flush_video(ignore_empty_transition=True)

        # Clear cache
        self._elapsed_steps = 0
        self._episode_id += 1
        self._episode_data = []
        self._episode_info = {}
        self._render_images = []

        reset_kwargs = copy.deepcopy(kwargs)
        obs, info = super().reset(**kwargs)

        if self.save_video:
            self._render_images.append(self.env.render())

        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1

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

    def close(self) -> None:
        if self.save_video:
            if self.save_on_reset:
                self.flush_video(ignore_empty_transition=True)
        return super().close()
