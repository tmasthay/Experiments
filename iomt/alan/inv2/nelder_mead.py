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
from colorama import Fore, Style
from scipy.optimize import minimize


def check_nans(u: torch.Tensor, *, name: str = 'output', msg: str = '') -> None:
    if torch.isnan(u).any():
        n_nans = torch.sum(torch.isnan(u))
        percent_nans = 100 * n_nans / u.numel()
        raise ValueError(
            'NaNs detected in'
            f' "{name}"\n{msg}\n{n_nans}/{u.numel()} ({percent_nans:.2f}%)'
        )


class SourceAmplitudes:
    def __init__(
        self,
        ny: int,
        nx: int,
        init_loc0: float,
        init_loc1: float,
        halfwidth: int,
        beta: float,
        source_trace: torch.Tensor,
        trainable: bool = True,
    ):
        self.ny = ny
        self.nx = nx
        self.loc = torch.tensor(
            [init_loc0, init_loc1],
            dtype=source_trace.dtype,
            device=source_trace.device,
        )
        self.source_trace = source_trace
        self.halfwidth = halfwidth
        self.beta = (
            beta
            if type(beta) == torch.Tensor
            else torch.tensor(
                beta, device=source_trace.device, dtype=source_trace.dtype
            )
        )

    def _get_weight(self, loc, n):
        x = (
            torch.arange(
                n,
                device=self.source_trace.device,
                dtype=self.source_trace.dtype,
            )
            - loc
        )
        bessel_arg = torch.relu(self.beta * (1 - (x / self.halfwidth) ** 2))
        bessel_term = torch.i0(bessel_arg) / torch.i0(self.beta) * torch.sinc(x)
        return bessel_term * torch.sinc(x)

    def forward(self, loc):
        return (
            self.source_trace[:, None]
            * self._get_weight(loc[0], self.ny).reshape(1, -1, 1, 1)
            * self._get_weight(loc[1], self.nx).reshape(1, 1, -1, 1)
        ).reshape(self.source_trace.shape[0], -1, self.source_trace.shape[-1])


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    stars = 80 * '*'
    print(
        Fore.GREEN
        + stars
        + '\nPreprocessing Config\n'
        + stars
        + Style.RESET_ALL
    )
    c = OmegaConf.to_container(cfg, resolve=True)
    return DotDict(c)


@hydra.main(config_path='all/old', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)

    if c.get('dupe', True):
        dupe(hydra_out('stream'), verbose=True, editor=c.get('editor', None))

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

    rec_loc = source_locations_all.view(c.n_shots, -1, 2)

    v = torch.ones(c.ny, c.nx).to(c.device) * c.vel
    v = v[: c.ny, : c.nx]

    source_amplitudes_true = (
        dw.wavelets.ricker(c.freq, c.nt, c.dt, c.peak_time_factor / c.freq)
        .repeat(c.n_shots, 1)
        .to(c.device)
    )

    source_amplitudes = SourceAmplitudes(
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

    def forward(loc):
        amps = source_amplitudes.forward(loc)
        check_nans(amps, name='amps')
        u = dw.scalar(
            v,
            [c.dy, c.dx],
            dt=c.dt,
            source_amplitudes=amps,
            source_locations=final_src_loc,
            receiver_locations=rec_loc,
            pml_width=c.pml_width,
            pml_freq=c.pml_freq,
        )[-1]
        check_nans(u, name='output')
        return u

    obs_data_true = forward(ref_amplitudes.loc)

    def loss_function(loc):
        loc_tensor = torch.tensor(
            loc,
            dtype=source_amplitudes_true.dtype,
            device=source_amplitudes_true.device,
        )
        obs_data = forward(loc_tensor)
        loss = torch.nn.functional.mse_loss(obs_data, obs_data_true)
        return loss.item()

    initial_guess = source_amplitudes.loc.cpu().numpy()
    result = minimize(
        loss_function,
        initial_guess,
        method='Nelder-Mead',
        options={'maxiter': c.n_epochs, 'disp': True},
    )

    optimized_loc = result.x
    print(f'Optimized Source Location: {optimized_loc}')
    print(f'True Source Location: {ref_amplitudes.loc.cpu()}')

    with open('.latest', 'w') as f:
        f.write(f'cd {hydra_out()}')

    print('To see the results of this run, run\n    . .latest')


if __name__ == "__main__":
    main()
