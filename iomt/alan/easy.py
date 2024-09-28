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
from mh.core import hydra_out, DotDict
from misfit_toys.swiffer import dupe


def check_nans(u: torch.Tensor, *, name: str = 'output', msg: str = '') -> None:
    if torch.isnan(u).any():
        # count number of NaNs
        n_nans = torch.sum(torch.isnan(u))
        percent_nans = 100 * n_nans / u.numel()
        raise ValueError(
            'NaNs detected in'
            f' "{name}"\n{msg}\n{n_nans}/{u.numel()} ({percent_nans:.2f}%)'
        )


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

    v = torch.ones(c.ny, c.nx).to(c.device) * c.vel
    v = v[: c.ny, : c.nx]

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

    obs_data_true = forward(amps=ref_amplitudes(), msg='True')

    optimizer = torch.optim.LBFGS(source_amplitudes.parameters(), lr=c.lr)
    loss = torch.nn.MSELoss()

    num_frames = 10
    save_freq = max(1, c.n_epochs // num_frames)
    # param_history = [source_amplitudes().detach().clone()
    param_history = []
    # loss_history = [
    #     loss(
    #         obs_data_true, forward(amps=source_amplitudes(), msg='Initial').detach()
    #     )
    # ]
    loss_history = []
    grad_norm_history = []
    for epoch in range(c.n_epochs):
        num_calls = 0

        def closure():
            nonlocal num_calls
            num_calls += 1
            optimizer.zero_grad()
            obs_data = forward(amps=source_amplitudes(), msg=f'{epoch=}')
            loss_val = loss(obs_data, obs_data_true)
            loss_val.backward()
            if num_calls == 1 and epoch % save_freq == 0:
                param_history.append(source_amplitudes())
                loss_history.append(loss_val)
            return loss_val

        loss_val = optimizer.step(closure)
        true_error = torch.norm(
            source_amplitudes.loc().detach() - ref_amplitudes.loc().detach()
        )
        print(
            f'Epoch {epoch}, Loss: {loss_val.item()}, neg_loc_grad:'
            f' {list(-source_amplitudes.loc.get_grad().cpu().numpy())}'
            f' loc: {list(source_amplitudes.loc().detach().cpu().numpy())}'
            f' true_error: {true_error}',
            flush=True,
        )

    with open('.latest', 'w') as f:
        f.write(f'cd {hydra_out()}')

    print('To see the results of this run, run\n    . .latest')


if __name__ == "__main__":
    main()
