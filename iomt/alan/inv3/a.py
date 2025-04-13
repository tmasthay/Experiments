# 
# @VS@ cd _dir && python -W ignore _file

import hydra
from time import time

start_load_torch  = time()
import torch
end_load_torch = time()
print(f"torch load time: {end_load_torch - start_load_torch:.5e} seconds")
import deepwave
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from mh.core import DotDict as DD, DotDictImmutable as DDI, exec_imports, exec_imports_constrained
from misfit_toys.utils import apply, apply_all

def preprocess_cfg(cfg: DictConfig):
    c = DD(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports_constrained(c)
    c = exec_imports(c)
    # c.simulation = apply(c.simulation, allow_implicit=True)
    apply_all(c, allow_implicit=True)
    return c

@hydra.main(config_path="all/a", config_name="default", version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    print(c)

if __name__ == "__main__":
    main()