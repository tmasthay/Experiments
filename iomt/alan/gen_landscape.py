# flake8: noqa
from copy import deepcopy
import os
from typing import Tuple
import numpy as np
import torch
import deepwave as dw
from misfit_toys.utils import (
    bool_slice,
    clean_idx,
    git_dump_info,
    runtime_reduce,
)
from mh.typlotlib import save_frames, get_frames_bool
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from misfit_toys.fwi.seismic_data import ParamConstrained, Param
import hydra
from omegaconf import OmegaConf, DictConfig
from dotmap import DotMap
from mh.core import (
    hydra_out,
    DotDict,
    set_print_options,
    torch_stats,
    exec_imports,
)
from misfit_toys.swiffer import dupe
from time import time, sleep
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

set_print_options(callback=torch_stats('all'))


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c = exec_imports(
        c,
        delim=c.import_specs.delim,
        import_key=c.import_specs.key,
        ignore_spaces=c.import_specs.ignore_spaces,
    )
    assert 'resolve_order' in c.keys()

    # You should refactor so that it is literally immutable but let's forgo that for now.
    init_resolve = deepcopy(c.resolve_order)
    for call_key, self_key in c.resolve_order:
        c = runtime_reduce(c, relax=True, call_key=call_key, self_key=self_key)
        assert init_resolve == c.resolve_order, "Resolve should not change"

    return c


@hydra.main(
    config_path='cfg/landscape/elastic/gen', config_name='cfg', version_base=None
)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    print(c)


if __name__ == "__main__":
    main()
