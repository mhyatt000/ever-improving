import os
import improve
import numpy as np
import mediapy as mp
from mediapy import write_video
from tqdm import tqdm
from PIL import Image
from improve.pac.qrdqn.qrdqn import load_data

import os
import os.path as osp
import warnings
import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import improve
import improve.config.prepare
import improve.config.resolver
from improve.sb3 import util
from improve.wrapper import residualrl as rrl
from improve.wrapper.flex import HDF5LoggerWrapper

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

VIDEO_PATH = os.path.join(improve.RESULTS, "rollouts/simpler_env")
EPISODES = 1
MAX_STEPS = 3

@hydra.main(config_path=improve.CONFIG, config_name="config")
def main(cfg):
    
    env = rrl.make(cfg.env)
    env = HDF5LoggerWrapper(env)
    
    # breakpoint()
    for episode_n in tqdm(range(EPISODES)):
        images = []
        obs, _ = env.reset()
        
        breakpoint()
        
        for step in tqdm(range(MAX_STEPS)):
            images.append(obs['simpler-img'])
            
            obs, reward, done, truncated, info = env.step(0)      #np.random.uniform(low=-0.01, high=0.01, size=(7)))
            
            if done or truncated:
                break
            
        # add terminated image
        images.append(obs['simpler-img'])
        write_video(f"{VIDEO_PATH}/episode_{episode_n}_success_{int(reward)}_test.mp4", images, fps=5)
        
    env.close()
            
    

if __name__ == "__main__":
    main()