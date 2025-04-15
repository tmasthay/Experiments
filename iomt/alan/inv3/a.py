#
# @VS@ cd _dir && python -W ignore _file

import hydra
from time import time

import omegaconf
import torch

import deepwave as dw
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from mh.core import (
    DotDict as DD,
    DotDictImmutable as DDI,
    exec_imports,
    exec_imports_constrained,
    TraceError
)
from misfit_toys.utils import apply, apply_all


def preprocess_cfg(cfg: DictConfig):
    try: 
        c = DD(OmegaConf.to_container(cfg, resolve=True))
    except omegaconf.errors.InterpolationKeyError as e:
        raise RuntimeError(
            f"Interpolation error: debug info below \n\n{DD(OmegaConf.to_container(cfg, resolve=False))}\n\n"
        ) from e
    c = exec_imports_constrained(c)
    c = exec_imports(c)
    # c.simulation = apply(c.simulation, allow_implicit=True)
    return c


@hydra.main(config_path="all/a", config_name="default", version_base=None)
def main(cfg: DictConfig):

    c = preprocess_cfg(cfg)
    
    for i,v in c.resolve_order.items():
        c.self_ref_resolve(self_key=f'self_{v.key}')
        c = apply_all(
            c, allow_implicit=True, call_key=f"__call_{v.key}__", relax=False
        )
        if v.check is not None:
            try:
                pass_check = v.check(c)
                if not pass_check:
                    raise RuntimeError(
                        f"Check {v.check} failed for {v.key}"
                    )
            except Exception as e:
                raise TraceError(
                    v.check,
                    f"Unexpected error happened during check {v.check} for key={v.key}"
                ) from e
                

    print(c)


if __name__ == "__main__":
    main()
