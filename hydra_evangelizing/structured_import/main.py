from mh.core import exec_imports_constrained, DotDict as DD, DotDictImmutable as DDI
import hydra
from omegaconf import DictConfig, OmegaConf

def preprocess_cfg(cfg: DictConfig) -> DD:
    c = DD(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports_constrained(c)
    return c

@hydra.main(config_path="all/main", config_name="default", version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    print(c)

if __name__ == "__main__":
    main()