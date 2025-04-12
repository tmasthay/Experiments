import hydra
import torch
import deepwave
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from mh.core import DotDict as DD, DotDictImmutable as DDI

def preprocessCfg(cfg: DictConfig):
    c = DD(OmegaConf.to_container)
    return c

@hydra.main(config_path="all/a", config_name="default", version_base=None)
def main(cfg: DictConfig):
    c = preprocessCfg(cfg)
    print(c)

if __name__ == "__main__":
    main()