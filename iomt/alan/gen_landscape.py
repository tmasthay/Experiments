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

# set_print_options(callback=torch_stats('all'))
set_print_options(callback=torch_stats(['shape']))


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
    c = exec_imports(
        c,
        delim=c.import_specs.delim,
        import_key=c.import_specs.key,
        ignore_spaces=c.import_specs.ignore_spaces,
    )

    # put any derivations of the config here but they cannot rely on the rt variables
    # These should only be of very basic type conversions or other simple operations
    # For example, if you have t0=0, dt=0.1, nt=100, then go ahead and calculate
    #     c.t = torch.linspace(c.t0, c.t0 + c.dt * (c.nt - 1), c.nt)
    # But ONLY these types of calculations should be done here! Anything more complex
    #    should be handled in the runtime part of the config "rt" section!
    c = runtime_reduce(
        c,
        relax=True,
        allow_implicit=True,
        exc=['rt', 'resolve_order'],
        call_key="__call_pre__",
        self_key="self_pre",
    )

    cfg_orig = deepcopy(c.filter(exclude=['rt', 'resolve_order', 'dep']).dict())
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

        non_rt_diff = dict_diff(
            c.filter(exclude=['rt', 'dep']).dict(), cfg_orig
        )
        assert non_rt_diff == {}, f'{c=}, {non_rt_diff=}, {cfg_orig=}'

    def bnd_assert(bounds, val, name):
        assert (
            bounds[0] < val.min()
        ), f'lower={bounds[0]}, {name}_min={val.min()}'
        assert (
            val.max() < bounds[1]
        ), f'{name}_max={val.max()}, upper={bounds[1]}'

    bnd_assert(c.bounds.vp, c.rt.vp, 'vp')
    bnd_assert(c.bounds.vs, c.rt.vs, 'vs')
    bnd_assert(c.bounds.rho, c.rt.rho, 'rho')

    # Assert that the vp/vs ratio is within bounds
    assert not (c.rt.vp < c.rt.vs).any()

    return c


@hydra.main(config_path='cfg_gen', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    try:
        c.rt.res = c.main.callback(c)
    except Exception as e:
        # print(f'Error: {e}')
        print(f'{c.main=}')
        raise e
    c.postprocess.callback(c, path=hydra_out())

    with open('.latest', 'w') as f:
        f.write(f'cd {hydra_out()}')

    print(f'\nRun . .latest to cd to the latest output directory\n')


if __name__ == "__main__":
    main()
