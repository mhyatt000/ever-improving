import json
import os
import time

import h5py
import numpy as np
from tqdm import tqdm

# Number of steps to simulate
num_steps = int(1e3)

# Create dummy data
observations = [np.random.random((84, 84, 3)) for _ in range(num_steps)]
rewards = [np.random.random() for _ in range(num_steps)]
dones = [False for _ in range(num_steps)]
actions = [np.random.randint(0, 2) for _ in range(num_steps)]
infos = [{"info_key": "info_value"} for _ in range(num_steps)]

# Benchmark HDF5
hdf5_filename = "benchmark.h5"
with h5py.File(hdf5_filename, "w") as f:
    step_group = f.create_group("steps")
    for i in tqdm(range(num_steps)):
        step_dataset = step_group.create_group(f"step_{i}")
        step_dataset.create_dataset("observation", data=observations[i])
        step_dataset.create_dataset("reward", data=np.array(rewards[i]))
        step_dataset.create_dataset("done", data=np.array(dones[i]))
        step_dataset.create_dataset("action", data=np.array(actions[i]))

        info_group = step_dataset.create_group("info")
        for key, value in infos[i].items():
            if isinstance(value, str):
                dtype = h5py.special_dtype(vlen=str)
                info_group.create_dataset(key, data=value, dtype=dtype)
            else:
                info_group.create_dataset(key, data=np.array(value))

start_time = time.time()
with h5py.File(hdf5_filename, "r") as f:
    step_group = f["steps"]
    for i in tqdm(range(num_steps)):
        step_dataset = step_group[f"step_{i}"]
        observation = step_dataset["observation"][:]
        reward = step_dataset["reward"][()]
        done = step_dataset["done"][()]
        action = step_dataset["action"][()]
        info_group = step_dataset["info"]
        info = {key: info_group[key][()] for key in info_group}
end_time = time.time()
print(f"HDF5 read time: {end_time - start_time:.4f} seconds")

# Benchmark .npy and JSON
output_dir = "benchmark_npy_json"
os.makedirs(output_dir, exist_ok=True)
for i in tqdm(range(num_steps)):
    np.save(os.path.join(output_dir, f"step_{i}_obs.npy"), observations[i])
    step_data = {
        "reward": rewards[i],
        "done": dones[i],
        "action": actions[i],
        "info": infos[i],
    }
    with open(os.path.join(output_dir, f"step_{i}_data.json"), "w") as f:
        json.dump(step_data, f)

start_time = time.time()
for i in tqdm(range(num_steps)):
    observation = np.load(os.path.join(output_dir, f"step_{i}_obs.npy"))
    with open(os.path.join(output_dir, f"step_{i}_data.json"), "r") as f:
        step_data = json.load(f)
    reward = step_data["reward"]
    done = step_data["done"]
    action = step_data["action"]
    info = step_data["info"]
end_time = time.time()
print(f".npy and JSON read time: {end_time - start_time:.4f} seconds")
