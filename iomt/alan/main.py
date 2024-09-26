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
n_shots = 1

# Put one source in each velocity model cell
source_locations_all = (
    torch.stack(
        torch.meshgrid((torch.arange(ny), torch.arange(nx)), indexing='ij'),
        dim=-1,
    )
    .repeat(n_shots, 1, 1)
    .int()
    .to(device)
)

# observe densely at specified depth 2

# rec_loc = (
#     torch.cartesian_prod(torch.tensor([2]), torch.arange(ny))
#     .repeat(n_shots, 1, 1)
#     .int()
#     .to(device)
# )
rec_loc = source_locations_all.detach().clone()
rec_loc = rec_loc.view(n_shots, -1, 2)
print(f'{rec_loc.shape=}')


nt = 1000
dt = 0.0004
freq = 25.0
peak_time = 1.5 / freq
dy, dx = 4.0, 4.0
v = torch.ones(ny, nx).to(device) * 1500.0
v[ny//2, :] = 3000.0
v[:, 3 * nx // 4] = 4500.0
v[3 * ny // 4, :] = 2000.0
v[:, 3 * ny // 4] = 4000.0
v = v[:ny, :nx]
beta = [4.0, 4.0]
halfwidth = [70, 70]

t = torch.linspace(0, nt * dt, nt, device=device)
source_amplitudes_true = (
    dw.wavelets.ricker(freq, nt, dt, peak_time).repeat(n_shots, 1).to(device)
)
source_amplitudes = SourceAmplitudes(
    ny=ny,
    nx=nx,
    init_loc0=100.5,
    init_loc1=60.6753435,
    source_trace=source_amplitudes_true,
    beta=beta[0],
    halfwidth=halfwidth[0],
)
# other_sources = SourceAmplitudes(
#     ny=ny,
#     nx=nx,
#     init_loc0=20.5,
#     init_loc1=50.4,
#     source_trace=source_amplitudes_true,
#     beta=beta[1],
#     halfwidth=halfwidth[1],
# )

# opts = dict(
#     cmap='seismic',
#     aspect='auto',
#     vmin=source_amplitudes_true.min(),
#     vmax=source_amplitudes_true.max(),
# )


# def plotter_amps(
#     *, data: torch.Tensor, idx: Tuple[slice, ...], fig: Figure, axes: Axes
# ):
#     plt.clf()
#     plt.imshow(data[idx], **opts)
#     plt.title(clean_idx(idx))
#     plt.colorbar()
#     plt.savefig(f'{idx[-1]}.jpg')


# tmp = source_amplitudes() + other_sources()
flat_amps = source_amplitudes()
# loop_amps = flat_amps.squeeze().reshape(ny, nx, -1)
# iter = bool_slice(*loop_amps.shape, none_dims=[0, 1], strides=[1, 1, 5])

# fig, axes = plt.subplots(1, 1)
# frames = get_frames_bool(
#     data=loop_amps.detach().cpu(),
#     iter=iter,
#     plotter=plotter_amps,
#     fig=fig,
#     axes=axes,
# )
# print(f'{len(frames)} frames')
# save_frames(frames, path='source_amplitudes.gif', verify_frame_count=True)


pml_width = 20
u = dw.scalar(
    v,
    [dy, dx],
    dt=dt,
    source_amplitudes=flat_amps,
    source_locations=source_locations_all.view(n_shots, -1, 2),
    receiver_locations=rec_loc,
    pml_width=pml_width,
)

# full_wavefield=u[0].detach().cpu().squeeze()
# physical_wavefield=full_wavefield[pml_width:-pml_width,pml_width:-pml_width]

# plt.imshow(physical_wavefield, **opts)
# plt.title('Final Wavefield')
# plt.ylabel('Depth')
# plt.xlabel('Offset')
# plt.colorbar()
# plt.savefig('final_wavefield.jpg')
# plt.clf()

# plt.imshow(u[-1].T.detach().cpu().squeeze(), **opts)
# plt.title('Observed Data')
# plt.ylabel('Time (s)')
# plt.xlabel('Offset Index')
# plt.savefig('obs_data.jpg')

obs_data = u[-1].detach().cpu().squeeze().view(ny, nx, -1)
opts = dict(
    cmap='seismic', aspect='auto', vmin=obs_data.min(), vmax=obs_data.max()
)


def plotter(*, data, idx, fig, axes):
    plt.clf()
    plt.imshow(data[idx], **opts)
    plt.imshow(v.detach().cpu(), cmap='nipy_spectral', alpha=0.3, aspect='auto')
    plt.colorbar()
    plt.ylabel('Depth')
    plt.xlabel('Offset')
    plt.title(clean_idx(idx))


iter = bool_slice(*obs_data.shape, none_dims=[0, 1], strides=[1, 1, 5])
frames = get_frames_bool(data=obs_data, iter=iter, plotter=plotter)
save_frames(frames, path='obs_data.gif')
