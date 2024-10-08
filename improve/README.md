# project contents

    .
    ├── cn                     # hydra config nodes
    ├── data                   # data processing and loading tools (not used much)
    ├── env                    # custom environments and env preprocessing
    ├── fm                     # tools for foundation model inference
    ├── hydra                  # hydra config resolver
    ├── logger                 # custom logger for sb3 => wandb
    ├── optim                  # rsqrt optimizer (sb3 does not like custom optimizers)
    │   └── schedule
    ├── sb3                    # stable-baselines3 algorithms and overrides
                               # most of repo the code is here
    │   ├── custom             # custom algorithms including residual policy
    │   │   └── rp_sac
    │   └── offline
    ├── utils                  # utility functions copied from other repos
                               # dont think these are used in main code, but kept for reference
    │   └── loss
    └── wrapper               # custom wrappers for SIMPLER => octo/RTX

    29 directories

# focused contents

pay special attention to these directories when contributing/reuse:

    .
    ├── cn                     # hydra config nodes
    ├── env                    # custom environments and env preprocessing
    ├── fm                     # tools for foundation model inference
    ├── hydra                  # hydra config resolver
    ├── sb3                    # stable-baselines3 algorithms and overrides
                               # most of repo the code is here
    │   ├── custom             # custom algorithms including residual policy
    │   │   └── rp_sac
    └── wrapper               # custom wrappers for SIMPLER => octo/RTX

