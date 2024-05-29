import hydra
from omegaconf import OmegaConf
import os
import os.path as osp

import improve
from pprint import pprint

ROOT = osp.dirname(osp.dirname(improve.__file__))
CONFIG = osp.join(ROOT, "config")


@hydra.main(config_path=CONFIG, config_name="config")
def main(cfg: OmegaConf) -> None:

    cfg = OmegaConf.to_container(cfg, resolve=True)
    pprint(cfg)
    # pprint(cfg['env'])


if __name__ == "__main__":
    main()
