# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import time

import hydra
import improve
import improve.config.prepare
import improve.config.resolver

import submitit

# log = logging.getLogger(__name__)


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def my_app(cfg):

    # env = submitit.JobEnvironment()
    print(f"Process ID {os.getpid()} executing task {cfg.task}, with {env}")
    time.sleep(1)


if __name__ == "__main__":
    my_app()
