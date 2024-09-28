import os
from typing import Tuple
import numpy as np
import torch
import deepwave as dw
from misfit_toys.utils import bool_slice, clean_idx
from mh.typlotlib import save_frames, get_frames_bool
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from misfit_toys.fwi.seismic_data import ParamConstrained, Param
import hydra
from omegaconf import OmegaConf, DictConfig
from dotmap import DotMap
from mh.core import hydra_out, DotDict, set_print_options, torch_stats
from misfit_toys.swiffer import dupe
from time import time

set_print_options(callback=torch_stats('all'))


def check_nans(u: torch.Tensor, *, name: str = 'output', msg: str = '') -> None:
    return u.nan_to_num(nan=0.0)


class SourceAmplitudes(torch.nn.Module):
    def __init__(
        self,
        *,
        ny: int,
        nx: int,
        init_loc0: float,
        init_loc1: float,
        halfwidth: int,
        beta: float,
        source_trace: torch.Tensor,
        trainable: bool = True,
    ):
        super().__init__()
        self.ny = ny
        self.nx = nx
        self.trainable = trainable
        if trainable:
            self.loc = ParamConstrained(
                p=torch.tensor([init_loc0, init_loc1]),
                minv=0,
                maxv=min(ny, nx),
                requires_grad=True,
            )
        else:
            self.loc = Param(
                p=torch.tensor([init_loc0, init_loc1]), requires_grad=False
            )
        self.source_trace = source_trace
        self.device = source_trace.device
        self.dtype = source_trace.dtype
        self.halfwidth = halfwidth
        self.beta = torch.tensor(beta).to(self.dtype).to(self.device)

    def _get_weight(self, loc, n):
        x = torch.arange(n, device=self.device, dtype=self.dtype) - loc
        # bessel_arg = self.beta * (1 - (x / self.halfwidth) ** 2).sqrt()
        bessel_arg = self.beta * (1 - (x / self.halfwidth) ** 2)
        bessel_arg = bessel_arg.nan_to_num(nan=0.0)
        bessel_term = torch.i0(bessel_arg) / torch.i0(self.beta) * torch.sinc(x)
        # input(bessel_term.max())
        # bessel_term[x.abs() > self.halfwidth] = 0.0
        return bessel_term * torch.sinc(x)

    def forward(self):
        loc = self.loc()
        return (
            self.source_trace[:, None]
            * self._get_weight(loc[0], self.ny).reshape(1, -1, 1, 1)
            * self._get_weight(loc[1], self.nx).reshape(1, 1, -1, 1)
        ).reshape(self.source_trace.shape[0], -1, self.source_trace.shape[-1])


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = OmegaConf.to_container(cfg, resolve=True)
    c = DotDict(c)
    c.peak_time = c.peak_time_factor / c.freq
    c.dtype = torch.float32
    return c


@hydra.main(config_path='cfg', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)

    def get_path(x='', ext='pt'):
        data_path = os.path.join(os.environ['CONDA_PREFIX'], 'data')
        rel_path = 'marmousi/deepwave_example/shots16'
        base = os.path.join(data_path, rel_path)
        return os.path.join(base, x.replace(ext, '') + f'.{ext}')

    with open(get_path('metadata', 'pydict')) as f:
        metadata = DotDict(eval(f.read()))

    meta_priority = ['ny', 'nx', 'nt', 'dx', 'dt', 'freq', 'peak_time_factor']
    for k, v in metadata.items():
        if k not in c.keys() or k in meta_priority:
            c[k] = v

    # print(c.pretty_str())

    mid_idx = metadata.n_shots // 2
    # mid_idx = slice(None)
    # vp = c.vel * torch.ones_like(torch.load(get_path('vp_init'))).to(c.device)
    vp = torch.load(get_path('vp_true')).to(c.device)
    src_amp_y = (
        torch.load(get_path('src_amp_y'))[mid_idx]
        .view(c.n_shots, 1, -1)
        .to(c.device)
    )
    src_loc = (
        torch.load(get_path('src_loc_y'))[mid_idx]
        .repeat(c.n_shots, 1, 1)
        .to(c.device)
    )
    obs_data_iomt: torch.Tensor = (
        torch.load(get_path('obs_data'))[mid_idx]
        .view(c.n_shots, -1, c.nt)
        .to(c.device)
    )
    rec_loc = (
        torch.load(get_path('rec_loc_y'))[mid_idx]
        .view(c.n_shots, -1, 2)
        .long()
        .to(c.device)
    )
    # vp = torch.load(get_path('vp_true')).to(c.device)
    # src_amp_y = torch.load(get_path('src_amp_y')).to(c.device)
    # src_loc = torch.load(get_path('src_loc_y')).to(c.device)
    # obs_data: torch.Tensor = torch.load(get_path('obs_data')).to(c.device)
    # rec_loc = torch.load(get_path('rec_loc_y')).long().to(c.device)

    print(c.pretty_str())

    for k, v in locals().items():
        if isinstance(v, torch.Tensor):
            print(f'{k=}, {v.shape=}, {v.dtype=}')

    def forward(y_idx, x_idx):
        loc = torch.tensor([y_idx, x_idx], device=c.device)
        loc = loc.repeat(c.n_shots, 1).view(c.n_shots, -1, 2)
        return dw.scalar(
            vp,
            c.dy,
            dt=c.dt,
            source_amplitudes=src_amp_y,
            source_locations=loc,
            receiver_locations=rec_loc,
            accuracy=8,
            pml_freq=c.freq,
        )[-1]

    # y_ref = src_loc[:, :, 0].item()
    # x_ref = src_loc[:, :, 1].item()
    y_ref = c.ny // 2
    x_ref = c.nx // 2
    obs_data = forward(y_ref, x_ref)

    def error(y_idx, x_idx):
        u = forward(y_idx, x_idx)
        return torch.nn.functional.mse_loss(u, obs_data).item()

    y_srcs = torch.arange(10, c.ny - 10)
    x_srcs = torch.arange(10, c.nx - 10)
    final = torch.empty(y_srcs.numel(), x_srcs.numel())
    start_time = time()
    for i, yy in enumerate(y_srcs):
        for j, xx in enumerate(x_srcs):
            final[i, j] = error(yy, xx)
            if j == 0:
                steps_remaining = y_srcs.numel() - i
                time_elapsed = time() - start_time
                time_per_step = time_elapsed / (i + 1)
                time_remaining = steps_remaining * time_per_step
                print(
                    f'({i}, {j}) ->'
                    f' {final[i, j].item()} [{time()-start_time:.2f}s elasped,'
                    f' {time_remaining:.2f}s remaining]'
                )

    torch.save(final, get_path('landscape'))


if __name__ == "__main__":
    main()
