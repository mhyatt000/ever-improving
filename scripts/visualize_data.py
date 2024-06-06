import os
import improve
import numpy as np
import mediapy as mp
from mediapy import write_video
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from improve.pac.qrdqn.qrdqn import load_data

VIDEO_PATH = os.path.join(improve.RESULTS, "rollouts")
EPISODES = 20
WIDTH, HEIGHT = 640, 480

def main():
    
    loader = load_data(batch_size=1)
    
    for episode_n in tqdm(range(EPISODES)):
        new_step = next(loader)
    
        images = []
        while not (new_step['terminated'].item() or new_step['truncated'].item()):
            # breakpoint()
            # images.append(Image.fromarray(new_step['observation']['simpler-img'][0].numpy()))
            plt.imshow(new_step['observation']['simpler-img'][0].numpy())
            plt.pause(0.1)
        
            # breakpoint()
            
            new_step = next(loader)
            
        # add terminated image
        # images.append(Image.fromarray(new_step['observation']['simpler-img'][0].numpy()))
        plt.imshow(new_step['observation']['simpler-img'][0].numpy())
        plt.pause(0.1)
        # write_video(f"{VIDEO_PATH}/episode_{episode_n}_success_{int(new_step['reward'].item())}_residual-acting_{int((new_step['action'] != 0).all().item())}.mp4", [np.array(image.resize((WIDTH, HEIGHT))) for image in images], fps=5)
        
        
            
    

if __name__ == "__main__":
    main()