import gymnasium as gym
import numpy as np
import simpler_env as simpler
from gymnasium import spaces
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name

# import hydra
# import improve
# from omegaconf import OmegaConf
# from omegaconf import OmegaConf as OC


class ReachTaskWrapper(gym.Wrapper):
    """
    makes the grasp task into the reach task
    Arguments:
        env: the environment to wrap
        use_sparse_reward: whether to use sparse reward or not
        thresh: the distance threshold to consider the task as successful
            larger is easier
        reward_clip: the maximum reward to clip the dense reward
    """

    def __init__(
        self,
        env,
        use_sparse_reward=True,
        thresh=0.1,
        reward_clip=0.5,
    ):
        super().__init__(env)

        self.use_sparse_reward = use_sparse_reward
        self.reward_clip = reward_clip
        self.thresh = thresh

        # assert env.obs_mode == "state_dict", "obs_mode must be state_dict"

    def step(self, action):
        obs, rew, success, truncated, info = self.env.step(action)
        rew, success, info = self.compute_success(obs, info)
        return obs, rew, success, truncated, info

    def compute_success(self, obs, info):

        # reached if abs distance between tcp and object is less than 1cm

        # old version only works for state_dict obs_mode
        # dist = np.abs(obs["extra_tcp_to_obj_pos"])

        dist = self.obj_wrt_eef()
        reached = dist < self.thresh
        reached = np.all(reached)

        info["success"] = reached
        info["reached"] = reached
        success = info["success"]

        if self.use_sparse_reward:
            rew = 1.0 if success else 0.0
        else:
            rew = np.clip(1 - np.tanh(10 * np.linalg.norm(dist)), -1, self.reward_clip)
            rew = rew if not success else 1.0

        return rew, success, info

    def get_tcp(self):
        """tool-center point, usually the midpoint between the gripper fingers"""
        eef = self.agent.config.ee_link_name
        tcp = get_entity_by_name(self.agent.robot.get_links(), eef)
        return tcp

    def obj_wrt_eef(self):
        """Get the object pose with respect to the end-effector frame"""
        return self.obj_pose.p - self.get_tcp().pose.p


# @hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
# def main(cfg):
def main():

    # from improve.wrapper.residualrl import rrl
    # assert cfg.get("sweep_id", None) == "debug"
    # assert ( cfg.env.foundation.name is None), f"cfg.env.foundation.name is not None: {cfg.env.foundation.name}"
    # pprint(OmegaConf.to_container(cfg, resolve=True))
    import warnings
    from pprint import pprint

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    """
    render_mode = "cameras"
    env = rrl.make(
        cfg.env,
        obs_mode="state_dict",  # "rgbd",
        render_mode=render_mode,
    )
    """

    env = simpler.make(
        "google_robot_pick_horizontal_coke_can",
        obs_mode="state_dict",
        max_episode_steps=60,
        render_mode="human",
    )

    env = ReachTaskWrapper(env, use_sparse_reward=False)
    obs, info = env.reset()
    print(obs)

    from tqdm import tqdm

    for _ in range(30):
        obs, info = env.reset(seed=1)
        done = False
        for t in tqdm(range(60)):
            action = env.action_space.sample()
            obs, rew, success, truncated, info = env.step(action)
            done = truncated or success
            if done:
                break
            print(rew)


if __name__ == "__main__":
    main()
