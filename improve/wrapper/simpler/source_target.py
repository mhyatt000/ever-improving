import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper


class SourceTargetWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space

        # add src/target object pose to observation space
        new_spaces = {
            "src-pose": 7,
            "tgt-pose": 7,
            "src-wrt-eef": 3,
            "tgt-wrt-eef": 3,
        }

        for space, dim in new_spaces.items():
            self.observation_space[space] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
            )

    def wrt_eef(self, obj_pose):
        """Get the object pose with respect to the end-effector frame"""
        return obj_pose.p - self.get_tcp().pose.p

    def observation(self, observation):
        # get src and target object pose
        src_pose, tgt_pose = self.source_obj_pose, self.target_obj_pose
        src_pose, tgt_pose = np.hstack((src_pose.p, src_pose.q)), np.hstack(
            (tgt_pose.p, tgt_pose.q)
        )

        observation["src-pose"] = src_pose
        observation["tgt-pose"] = tgt_pose
        observation["src-wrt-eef"] = self.objs_wrt_eef(self.source_obj_pose)
        observation["tgt-wrt-eef"] = self.objs_wrt_eef(self.target_obj_pose)

        return observation

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, success, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, success, truncated, info
