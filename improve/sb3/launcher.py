import json

from hydra.core.hydra_config import HydraConfig
import os
import os.path as osp
import subprocess
import time
from pprint import pprint

import hydra
import improve
from improve.config import resolver
from omegaconf import OmegaConf as OC


def list_gpus():
    """List all the GPUs on the machine."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        gpus = []
        for line in output.decode("utf-8").split("\n"):
            if line.strip():
                index, name, total_memory, used_memory = line.split(",")
                gpus.append(
                    {
                        "index": int(index),
                        "name": name.strip(),
                        "total_memory": int(total_memory),
                        "used_memory": int(used_memory),
                    }
                )
        return gpus
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        return []


def select_free_gpu(gpus, memory_threshold=500):
    """Select a GPU that is free (memory usage below the threshold)."""
    for gpu in gpus:
        if gpu["used_memory"] < memory_threshold:
            return gpu["index"]
    return None


def _run(cmd, wait=True):

    if wait:
        return os.system(cmd)
    else:
        # Execute the command without waiting for it to complete
        process = subprocess.Popen(command, shell=True)
        return process


def run_script_on_free_gpu(script_path, script_options):
    """Run a script with specified options on a free GPU."""
    gpus = list_gpus()
    if not gpus:
        print("No GPUs found or error querying GPUs.")
        return

    free_gpu = select_free_gpu(gpus)
    if free_gpu is None:
        print("No free GPU available.")
        return

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
    print(f"Using GPU {free_gpu}")

    # Build the command to run the script
    command = f"python {script_path} {script_options}"
    print(f"Running command: {command}")

    _run(command)


@hydra.main(config_path=improve.CONFIG, config_name="config")
def main(cfg):
    base = HydraConfig.get()

    pprint(OC.to_container(base, resolve=True))  # keep after wandb so it logs
    # print(cfg.hydra.sweeper)
    quit()

    _main = osp.join(osp.dirname(__file__), "main.py")
    script_options = "--option1 value1 --option2 value2"
    run_script_on_free_gpu(_main, script_options)


if __name__ == "__main__":
    main()
