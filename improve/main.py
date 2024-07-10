import hydra
from omegaconf import OmegaConf
import os
import os.path as osp

import improve
from pprint import pprint
import improve.hydra.resolver

@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg: OmegaConf) -> None:

    cfg = OmegaConf.to_container(cfg, resolve=True)

    pprint(cfg['env'])
    # pprint(cfg)
    print('\n\n\n')

    print(cfg['algo'])

if __name__ == "__main__":
    main()
