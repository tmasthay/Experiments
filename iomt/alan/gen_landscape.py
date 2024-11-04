# flake8: noqa
from copy import deepcopy
import os
import signal
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
from misfit_toys.swiffer import dupe, sco
from time import time, sleep
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from os.path import join as pj
import warnings
import yaml

# set_print_options(callback=torch_stats('all'))
set_print_options(callback=torch_stats(['shape']))


def warning_filter(message, category, filename, lineno, file=None, line=None):
    if "At least six grid cells per wavelength" in str(
        message
    ):  # Check for the specific message
        raise category(
            message
        )  # Raise an exception for the problematic warning


# Apply the custom warning filter
warnings.showwarning = warning_filter


def get_last_run_dir():
    try:
        cmd = ' | '.join(
            [
                """find "$(pwd)" -mindepth 1 -maxdepth 5  -type f -name "__COMPLETE__" -exec stat --format='%y %n' {} +""",
                'sort',
                'tail -n 1',
            ]
        )
        res1 = sco(cmd, split=False)
        res2 = res1.split()[-1].replace('/.hydra', '').split("/")[:-1]
        res = '/'.join(res2)
        return res
    except Exception as e:
        return None


def input_with_timeout(prompt, default_val, timeout_time):
    def timeout_handler(signum, frame):
        raise TimeoutError

    # Set up the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)

    try:
        # Start the timer with the given timeout time
        signal.alarm(timeout_time)
        user_input = input(prompt)
        # Cancel the alarm if input is received in time
        signal.alarm(0)
        return user_input
    except TimeoutError:
        return default_val


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
    prev_data_dir = cfg.prev_data_dir or get_last_run_dir()
    if prev_data_dir is None:
        use_prev_data = False
    else:
        use_prev_data = cfg.use_prev_data
    if use_prev_data:
        # reload the config from the previous run
        c2 = OmegaConf.to_container(
            OmegaConf.load(pj(prev_data_dir, '.hydra', 'config.yaml')),
            resolve=True,
        )
        c1 = OmegaConf.to_container(cfg, resolve=True)

        precedence = c1.get('precedence', ['postprocess'])

        c = {}
        for k, v in c1.items():
            if k in precedence or k not in c2:
                c[k] = v
        for k, v in c2.items():
            if k not in precedence or k not in c:
                c[k] = v
        c = DotDict(c)

        # rewrite the config.yaml, overrides.yaml, and hydra.yaml
        # to the new output directory
        for f in ['config.yaml', 'overrides.yaml', 'hydra.yaml']:
            ref = pj(prev_data_dir, '.hydra', f)
            new_copy = pj(hydra_out(), '.hydra', f)
            os.system(f'cp {ref} {new_copy}')
    else:
        c = DotDict(OmegaConf.to_container(cfg, resolve=True))
    c.use_prev_data = use_prev_data
    c.prev_data_dir = prev_data_dir
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

    with open(hydra_out('.hydra/runtime_pre.yaml'), 'w') as f:
        yaml.dump(c.dict(), f)

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

    c.assert_keys_present(
        [
            'rt.data.vp',
            'rt.data.src_loc',
            'rt.data.rec_loc',
            'rt.data.src_amp',
            'rt.data.vs',
            'rt.data.rho',
        ]
    )

    def bnd_assert(bounds, val, name):
        assert (
            bounds[0] < val.min()
        ), f'lower={bounds[0]}, {name}_min={val.min()}'
        assert (
            val.max() < bounds[1]
        ), f'{name}_max={val.max()}, upper={bounds[1]}'

    bnd_assert(c.bounds.vp, c.rt.data.vp, 'vp')
    bnd_assert(c.bounds.vs, c.rt.data.vs, 'vs')
    bnd_assert(c.bounds.rho, c.rt.data.rho, 'rho')

    # Assert that the vp/vs ratio is within bounds
    vp_vs = c.rt.data.vp / (1e-6 + c.rt.data.vs)
    assert not torch.isnan(vp_vs).any(), f'{vp_vs=}'
    assert vp_vs.min() >= torch.sqrt(
        torch.tensor(2.0)
    ), f'{vp_vs.min().item()=}'

    return c


def read_prev_data(c: DotDict, path: str) -> DotDict:
    # get all absolute paths to pytorch files in "path"
    all_files = sco(f'find {path} -type f -name "*.pt"')
    keys = [
        '.'.join(f.replace('.pt', '').split('/')[-1].split('___'))
        for f in all_files
    ]
    for k, f in zip(keys, all_files):
        c[k] = torch.load(f)
    return c


@hydra.main(config_path='cfg_gen', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    if cfg.get('dupe', True):
        dupe(hydra_out('stream'), editor=cfg.get('editor', None))

    with open(hydra_out('git_info.txt'), 'w') as f:
        f.write(git_dump_info())

    c = preprocess_cfg(cfg)

    if not c.use_prev_data:
        try:
            ref_size = (1000, 1000, 1000)
            ref_time_per_forward_solve = 0.33

            num_forward_solves = c.src.n_horz * c.src.n_deep
            size_factor = (
                (c.grid.nx / ref_size[0])
                * (c.grid.ny / ref_size[1])
                * (c.grid.nt / ref_size[2])
            )
            estimated_time = (
                ref_time_per_forward_solve * num_forward_solves * size_factor
            )

            print(
                'Estimated time for forward solves:'
                f' {estimated_time:.2f} seconds'
            )

            # five minute cutoff before we actually query. Else just run.
            cutoff = torch.inf
            absolute_cutoff = 86400
            if estimated_time > absolute_cutoff:
                raise RuntimeError(
                    f"Estimated time is {estimated_time:.2f} seconds...>"
                    f" {absolute_cutoff} seconds. Exiting..."
                )
            if (
                not c.get('dupe', False)
                and estimated_time > cutoff
                and input_with_timeout('Continue? [y/n] y: ', 'y', 3).lower()
                != 'y'
            ):
                print('Exiting...')
                return
            begin_time = time()
            c.rt.data.res = c.main.callback(c)
            total_time = time() - begin_time
        except Exception as e:
            # print(f'Error: {e}')
            print(f'{c.main=}')
            raise e
    else:
        if c.prev_data_dir is None:
            c.prev_data_dir = get_last_run_dir()
        print(f'Loading previous data from run at\n    {c.prev_data_dir}')
        c = read_prev_data(c, path=c.prev_data_dir)

    # always callback the postprocessing even if we used previous data
    c = runtime_reduce(
        c,
        call_key='__call_post__',
        self_key='self_post',
        allow_implicit=True,
        relax=False,
    )
    c.postprocess.callback(c, path=hydra_out())

    with open('.latest', 'w') as f:
        f.write(f'cd {hydra_out()}')

    with open(f'{pj(hydra_out(), "__COMPLETE__")}', 'w') as f:
        f.write('-- RUN FINISHED --\n    This file is simply a placeholder')

    print(f'\nRun . .latest to cd to the latest output directory\n')

    if not c.use_prev_data:
        print(f'Run time of main callback: {total_time:.2f} seconds')


if __name__ == "__main__":
    main()
