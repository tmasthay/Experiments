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
    DotDictImmutable,
    set_print_options,
    torch_stats,
    exec_imports,
)
from misfit_toys.swiffer import dupe
from time import time, sleep
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

set_print_options(callback=torch_stats('all'))


def dict_diff(d1, d2, *, name1='d1', name2='d2'):
    if isinstance(d1, DotDict):
        d1 = d1.dict()
    if isinstance(d2, DotDict):
        d2 = d2.dict()
    diff = {}

    # Take the union of the keys in both dictionaries
    all_keys = d1.keys() | d2.keys()

    for key in all_keys:
        if key not in d1:
            diff[key] = [f'Only in {name1}', d2[key]]
        elif key not in d2:
            diff[key] = [f'Only in {name2}', d1[key]]
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            # Recursively handle nested dictionaries
            nested_diff = dict_diff(d1[key], d2[key], name1=name1, name2=name2)
            if nested_diff:
                diff[key] = nested_diff
        elif d1[key] != d2[key]:
            diff[key] = ['Different Value', d1[key], d2[key]]

    return diff


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c.rt = c.get('rt', None)
    cfg_orig = deepcopy(c.filter(exclude=['rt', 'resolve_order']).dict())
    resolve_order = deepcopy(c.resolve_order or [])
    del c.resolve_order
    c = exec_imports(
        c,
        delim=c.import_specs.delim,
        import_key=c.import_specs.key,
        ignore_spaces=c.import_specs.ignore_spaces,
    )

    for call_key, self_key in resolve_order:
        c = runtime_reduce(
            c,
            relax=True,
            call_key=call_key,
            self_key=self_key,
            allow_implicit=True,
        )

        non_rt_diff = dict_diff(c.filter(exclude=['rt']).dict(), cfg_orig)
        assert non_rt_diff == {}, f'{c=}, {non_rt_diff=}, {cfg_orig=}'

    return c


@hydra.main(
    config_path='cfg/landscape/elastic/gen',
    config_name='cfg',
    version_base=None,
)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    print(c)


if __name__ == "__main__":
    main()
