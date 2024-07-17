import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper
import improve.hydra.resolver

class DrawerWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        
        # add the drawer position and drawer-wrt-eef to the observation space
        self.observation_space["drawer-pose"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space["drawer-pose-wrt-eef"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        
    def drawer_wrt_eef(self):
        """Get the drawer pose with respect to the end-effector frame"""
        return self.drawer_pose.p - self.get_tcp().pose.p
   
    def observation(self, observation):
        drawer_pos = self.drawer.pose
        drawer_pos = np.hstack((drawer_pos.p, drawer_pos.q))
        
        observation["drawer-pose"] = drawer_pos
        observation["drawer-pose-wrt-eef"] = self.drawer_wrt_eef()
    
    def reset(self, **kwargs):  
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info
   
    def step(self, action):
        obs, reward, success, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, success, truncated, info
    
import hydra
@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):
    print(cfg)

    import simpler_env as simpler
    from improve.env import make_env
    env = make_env(cfg)()
    env = DrawerWrapper(env)

    obs, info = env.reset()
    print(obs.keys())

    obs, reward, success, truncated, info = env.step(env.action_space.sample())
    print("drawer pose", obs["drawer-pose"])
    print("drawer pose wrt eef", obs["drawer-pose-wrt-eef"])

if __name__ == "__main__":
    main()
