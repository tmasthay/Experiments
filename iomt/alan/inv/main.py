from typing import Tuple
import numpy as np
import torch
import deepwave as dw
from misfit_toys.utils import bool_slice, clean_idx, git_dump_info
from mh.typlotlib import save_frames, get_frames_bool
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from misfit_toys.fwi.seismic_data import ParamConstrained, Param
import hydra
from omegaconf import OmegaConf, DictConfig
from dotmap import DotMap
from mh.core import hydra_out, DotDict
from misfit_toys.swiffer import dupe
from helpers import EasyW1Loss
from scipy.optimize import minimize


def check_nans(u: torch.Tensor, *, name: str = 'output', msg: str = '') -> None:
    if torch.isnan(u).any():
        # count number of NaNs
        n_nans = torch.sum(torch.isnan(u))
        percent_nans = 100 * n_nans / u.numel()
        raise ValueError(
            'NaNs detected in'
            f' "{name}"\n{msg}\n{n_nans}/{u.numel()} ({percent_nans:.2f}%)'
        )


class SourceAmplitudesLegacy(torch.nn.Module):
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
            # self.loc = ParamConstrained(
            #     p=torch.tensor([init_loc0, init_loc1]),
            #     minv=0,
            #     maxv=min(ny, nx),
            #     requires_grad=True,
            # )
            self.loc = Param(
                p=torch.tensor([init_loc0, init_loc1]), requires_grad=True
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
        # input(f'{loc()=}, {n=}')
        x = torch.arange(n, device=self.device, dtype=self.dtype) - loc
        # bessel_arg = self.beta * (1 - (x / self.halfwidth) ** 2).sqrt()
        bessel_arg = torch.zeros(x.shape, device=self.device, dtype=self.dtype)
        idx = x.abs() <= self.halfwidth
        bessel_arg[idx] = (
            self.beta * (1 - (x[idx] / self.halfwidth) ** 2).sqrt()
        )

        assert not torch.isnan(bessel_arg).any()
        # input(f'{bessel_arg.max()}, {bessel_arg.min()}, {torch.i0( bessel_arg).max()}, {torch.i0(bessel_arg).min()}, {torch.i0(self.beta)}')
        # bessel_arg = self.beta * (1 - (x / self.halfwidth) ** 2)
        # bessel_arg = bessel_arg.nan_to_num(nan=0.0)
        bessel_term = torch.i0(bessel_arg) / torch.i0(self.beta)

        assert not torch.sinc(x).isnan().any(), f'{x.min()}, {x.max()}'
        assert not bessel_term.isnan().any()
        # input(bessel_term.max())
        # bessel_term[x.abs() > self.halfwidth] = 0.0
        return bessel_term * torch.sinc(x)

    def forward(self):
        loc = self.loc()
        # input(f'{loc[0].item()=}, {loc[1].item()=}')
        return (
            self.source_trace[:, None]
            * self._get_weight(loc[0], self.ny).reshape(1, -1, 1, 1)
            * self._get_weight(loc[1], self.nx).reshape(1, 1, -1, 1)
        ).reshape(self.source_trace.shape[0], -1, self.source_trace.shape[-1])


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
            self.loc = Param(
                p=torch.tensor([init_loc0, init_loc1]), requires_grad=True
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
        # Linear interpolation now
        x = torch.arange(n, device=self.device, dtype=self.dtype) - loc

        spectral_interp = torch.sinc(x)

        dist_weights = torch.zeros(
            x.shape, device=self.device, dtype=self.dtype
        )
        idx = x.abs() <= self.halfwidth
        dist_weights[idx] = 1 - (x[idx].abs() / self.halfwidth) ** 0.5

        # return spectral_interp * dist_weights
        return dist_weights

    def forward(self):
        loc = self.loc()
        # input(f'{loc[0].item()=}, {loc[1].item()=}')
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


class MyL2Loss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.mse_loss(x, self.target)


@hydra.main(config_path='cfg', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)

    if c.get('dupe', True):
        dupe(hydra_out('stream'), verbose=True, editor=c.get('editor', None))

    # Put one source in each velocity model cell
    source_locations_all = (
        torch.stack(
            torch.meshgrid(
                (torch.arange(c.ny), torch.arange(c.nx)), indexing='ij'
            ),
            dim=-1,
        )
        .repeat(c.n_shots, 1, 1)
        .int()
        .to(c.device)
    )

    rec_loc = source_locations_all.detach().clone()
    # rec_loc = source_locations_all[:, :c.ny, :].detach().clone()
    rec_loc = rec_loc.view(c.n_shots, -1, 2)
    # print(f'{rec_loc.shape=}')

    # v = torch.ones(c.ny, c.nx).to(c.device) * c.vel
    # v = v[: c.ny, : c.nx]
    v = torch.rand(c.ny, c.nx).to(c.device) * c.vel

    t = torch.linspace(0, c.nt * c.dt, c.nt, device=c.device)
    source_amplitudes_true = (
        dw.wavelets.ricker(c.freq, c.nt, c.dt, c.peak_time)
        .repeat(c.n_shots, 1)
        .to(c.device)
    )
    source_amplitudes: SourceAmplitudes = SourceAmplitudes(
        ny=c.ny,
        nx=c.nx,
        init_loc0=c.init_loc[0] * c.ny,
        init_loc1=c.init_loc[1] * c.nx,
        source_trace=source_amplitudes_true,
        beta=c.beta[0],
        halfwidth=c.halfwidth[0],
        trainable=True,
    )

    ref_amplitudes = SourceAmplitudes(
        ny=c.ny,
        nx=c.nx,
        init_loc0=c.ref_loc[0] * c.ny,
        init_loc1=c.ref_loc[1] * c.nx,
        source_trace=source_amplitudes_true,
        beta=c.beta[0],
        halfwidth=c.halfwidth[0],
        trainable=False,
    )

    final_src_loc = source_locations_all.view(c.n_shots, -1, 2)

    def forward(*, amps, msg=''):
        check_nans(amps, name='amps', msg=msg)
        u = dw.scalar(
            v,
            [c.dy, c.dx],
            dt=c.dt,
            source_amplitudes=amps,
            source_locations=final_src_loc,
            receiver_locations=rec_loc,
            pml_width=c.pml_width,
        )[-1]

        # throw error if a               NaNs are detected
        check_nans(u, name='output', msg=msg)

        return u

    # input(ref_amplitudes().cpu().shape)
    obs_data_true = forward(amps=ref_amplitudes(), msg='True')

    # optimizer = torch.optim.LBFGS(source_amplitudes.parameters(), lr=c.lr)
    # optimizer = NelderMeadOptimizer(source_amplitudes.parameters(), lr=c.lr)
    # optimizer = torch.optim.Adam(source_amplitudes.parameters(), lr=c.lr)
    # loss = torch.nn.MSELoss()
    loss = MyL2Loss(obs_data_true)
    # loss = EasyW1Loss(
        # obs_data_true, renorm=torch.nn.Softplus(beta=1, threshold=20)
    # )

    def my_function(loc):
        u = SourceAmplitudes(
            ny=c.ny,
            nx=c.nx,
            init_loc0=loc[0] * c.ny,
            init_loc1=loc[1] * c.nx,
            source_trace=source_amplitudes_true,
            beta=c.beta[0],
            halfwidth=c.halfwidth[0],
            trainable=True,
        )
        obs_data = forward(amps=u(), msg='True')
        return loss(obs_data).sum().item()

    result = minimize(
        my_function,
        [c.init_loc[0], c.init_loc[1]],
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'disp': True},
        callback=lambda x: print(x),
    )

    # with open('.latest', 'w') as f:
    #     f.write(f'cd {hydra_out()}')

    # print('To see the results of this run, run\n    . .latest')

    print("Optimization Result:", result)


if __name__ == "__main__":
    main()
