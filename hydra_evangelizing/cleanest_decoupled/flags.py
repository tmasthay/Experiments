import hydra
from mh.core import convert_dictconfig, DotDict
from omegaconf import DictConfig
import yaml


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    return c


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    # eat up chosen since it is only a dummy to simplify hydra config
    s = yaml.dump(preprocess_cfg(cfg.chosen))
    s = s.replace("!!python/object:mh.core.DotDict", "")
    print('\n'.join(s.split('\n')[1:-1]))


if __name__ == "__main__":
    main()
