# ever-improving

actively being developed

## Table of Contents

* [Description](#description)
* [Installation](#installation)
* [Usage](#usage)


# Description

```
.
├── config
│   ├── cleanrl
│   └── experiment
└── improve
    ├── cleanrl
    ├── purejaxrl
    │   └── experimental
    │       └── s5
    ├── simpler_mod
    └── wrappers
```

# Installation

* install SIMPLER
* install Octo
* `piip install -e .`

# Usage

### Using Hydra Tips:

`python main.py job/wandb=dont algo=sac train.use_train=True`

* a/b selects config group b from parent group a
* a.b sets attribute b of group a

`python main.py -m +exp=may29_sac`

* `-m` flag is needed for multirun experiments

### Experiments:

* SB3
    * `python improve/sb3/main.py`
* CleanRL
    * `python improve/cleanrl/ppo.py`

# License

TBD

# Contributing

TBD

# Acknowledgements

TBD

# References

TBD

