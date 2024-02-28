import hydra
from mh.core import convert_dictconfig, DotDict
from omegaconf import DictConfig
import yaml


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    return c


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    print(yaml.dump(c))


if __name__ == "__main__":
    main()
