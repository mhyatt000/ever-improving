import json
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

file_path = "start_infos.json"
with open(file_path, "r") as file:
    data = json.load(file)

pos_dict = defaultdict(list)
successes = []

for episode in data:
    pos_dict["source"].append(episode["episode_source_obj_init_pose_wrt_robot_base"])
    pos_dict["target"].append(episode["episode_target_obj_init_pose_wrt_robot_base"])
    successes.append(episode["success"])

subtitles = ["x", "y", "z", "a", "b", "c", "d"]
for position in ["source", "target"]:
    fig, axs = plt.subplots(7, figsize=(10, 40))
    fig.suptitle(f"{position} positions")

    for i in range(7):
        axs[i].hist(
            [
                pos_dict[position][j][i]
                for j in range(len(pos_dict[position]))
                if not successes[j]
            ],
            color="orange",
            alpha=0.5,
        )
        axs[i].hist(
            [
                pos_dict[position][j][i]
                for j in range(len(pos_dict[position]))
                if successes[j]
            ],
            alpha=0.5,
        )
        axs[i].yaxis.set_major_locator(ticker.MultipleLocator(2))
        axs[i].set_title(subtitles[i])

    plt.subplots_adjust(hspace=0.5)

    plt.savefig(f'env_info_{position}.png')
