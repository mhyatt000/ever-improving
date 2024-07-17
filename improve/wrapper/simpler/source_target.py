import gymnasium as gym
import numpy as np
import simpler_env as simpler
from gymnasium.core import Wrapper

import hydra
import improve
import improve.hydra.resolver


class SourceTargetWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        # add src/target object pose to observation space
        self.observation_space["src-pose"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space["tgt-pose"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space["src-pose-wrt-eef"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space["tgt-pose-wrt-eef"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def objs_wrt_eef(self, obj_pose):
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

        # calculate the distance wrt to eef
        observation["src-pose-wrt-eef"] = self.objs_wrt_eef(self.source_obj_pose)
        observation["tgt-pose-wrt-eef"] = self.objs_wrt_eef(self.target_obj_pose)

        return observation

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, success, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, success, truncated, info


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):
    print(cfg)

    from improve.env import make_env

    # multi_obj_envs = []
    for task in simpler.ENVIRONMENTS:
        cfg.env.foundation.task = task
        env = make_env(cfg)()
        
        if not env:
            print("Skipping", task)

        final_subtask = env.is_final_subtask()
        if final_subtask:
            pass
        else:
            print("Current task", env.get_language_instruction())
            
        while not final_subtask:
            env.advance_to_next_subtask()
            final_subtask = env.is_final_subtask()
            print("\tsubtask", env.get_language_instruction())

        # try:
        #     getattr(env, 'source_obj_pose')
        #     multi_obj_envs.append(task)
        #     print(f"Task {task} is a multi-obj environment")
        # except:
        #     print(f"Task {task} is not a multi-obj environment")

        env.close()

    # print(multi_obj_envs)

    # import json

    # # with open("multi_obj_envs.json", "w") as f:
    # #     json.dump(multi_obj_envs, f, indent=4)

    # # env = simpler.make(cfg.env.task)
    # env = make_env(cfg)()
    # env = SourceTargetWrapper(env)

    # obs, info = env.reset()
    # print(obs.keys())

    # obs, reward, success, truncated, info = env.step(env.action_space.sample())
    # print("src-pose", obs['src-pose'])
    # print("tgt-pose", obs['tgt-pose'])
    # print("src-pose-wrt-eef", obs['src-pose-wrt-eef'])
    # print("tgt-pose-wrt-eef", obs['tgt-pose-wrt-eef'])


if __name__ == "__main__":
    main()
