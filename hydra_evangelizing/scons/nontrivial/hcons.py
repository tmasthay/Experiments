import hydra
from omegaconf import DictConfig, OmegaConf
import os
import yaml

@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    c = OmegaConf.to_container(cfg, resolve=True)
    yaml.dump(c, open('input.yaml', 'w'))


if __name__ == "__main__":
    main()
