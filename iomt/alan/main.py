from typing import Tuple
import numpy as np
import torch
import deepwave as dw
from misfit_toys.utils import bool_slice, clean_idx
from mh.typlotlib import save_frames, get_frames_bool
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
    ):
        super().__init__()
        self.ny = ny
        self.nx = nx
        self.loc0 = torch.nn.Parameter(torch.tensor(float(init_loc0)))
        self.loc1 = torch.nn.Parameter(torch.tensor(float(init_loc1)))
        self.source_trace = source_trace
        self.device = source_trace.device
        self.dtype = source_trace.dtype
        self.halfwidth = halfwidth
        self.beta = torch.tensor(beta).to(self.dtype).to(self.device)

    def _get_weight(self, loc, n):
        x = torch.arange(n, device=self.device, dtype=self.dtype) - loc
        bessel_arg = self.beta * (1 - (x / self.halfwidth) ** 2).sqrt()
        bessel_term = torch.i0(bessel_arg) / torch.i0(self.beta) * torch.sinc(x)
        bessel_term[x.abs() > self.halfwidth] = 0.0
        return bessel_term * torch.sinc(x)

    def forward(self):
        return (
            self.source_trace[:, None]
            * self._get_weight(self.loc0, self.ny).reshape(1, -1, 1, 1)
            * self._get_weight(self.loc1, self.nx).reshape(1, 1, -1, 1)
        ).reshape(self.source_trace.shape[0], -1, self.source_trace.shape[-1])


device = 'cuda:0'
# Size of velocity model
ny, nx = 120, 130

# Put one source in each velocity model cell
source_locations_all = torch.stack(
    torch.meshgrid((torch.arange(ny), torch.arange(nx)), indexing='ij'), dim=-1
)

n_shots = 1

nt = 1000
dt = 0.0004
freq = 25.0
peak_time = 1.5 / freq
t = torch.linspace(0, nt * dt, nt, device=device)
source_amplitudes_true = (
    dw.wavelets.ricker(freq, nt, dt, peak_time).repeat(n_shots, 1).to(device)
)
beta = [4.0, 4.0]
halfwidth = [70, 70]
source_amplitudes = SourceAmplitudes(
    ny=ny,
    nx=nx,
    init_loc0=1.0,
    init_loc1=4.0,
    source_trace=source_amplitudes_true,
    beta=beta[0],
    halfwidth=halfwidth[0],
)
other_sources = SourceAmplitudes(
    ny=ny,
    nx=nx,
    init_loc0=20.5,
    init_loc1=50.4,
    source_trace=source_amplitudes_true,
    beta=beta[1],
    halfwidth=halfwidth[1],
)


def plotter_amps(
    *,
    data: torch.Tensor,
    idx: Tuple[slice, ...],
    fig: Figure,
    axes: Axes,
    opts: dict,
):
    print(clean_idx(idx))
    plt.clf()
    plt.imshow(data[idx], **opts)
    plt.colorbar()


# tmp = source_amplitudes() + other_sources()
tmp = other_sources() + source_amplitudes()
loop_amps = tmp.squeeze().reshape(ny, nx, -1)
torch.save(loop_amps, 'loop_amps.pt')
# input(loop_amps.shape)
iter = bool_slice(*loop_amps.shape, none_dims=[0, 1], strides=[1, 1, nt // 10])
opts = dict(
    cmap='seismic',
    aspect='auto',
    vmin=source_amplitudes_true.min(),
    vmax=source_amplitudes_true.max(),
)
frames = get_frames_bool(
    data=loop_amps.detach().cpu(), iter=iter, plotter=plotter_amps, opts=opts
)
# input(len(frames))
save_frames(frames, path='source_amplitudes.gif')
