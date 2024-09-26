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
        bessel_arg = self.beta * (1 - (x / self.halfwidth) ** 2).sqrt()
        bessel_term = torch.i0(bessel_arg) / torch.i0(self.beta) * torch.sinc(x)
        bessel_term[x.abs() > self.halfwidth] = 0.0
        return bessel_term * torch.sinc(x)

    def forward(self):
        loc = self.loc()
        return (
            self.source_trace[:, None]
            * self._get_weight(loc[0], self.ny).reshape(1, -1, 1, 1)
            * self._get_weight(loc[1], self.nx).reshape(1, 1, -1, 1)
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

rec_loc = source_locations_all.detach().clone()
rec_loc = rec_loc.view(n_shots, -1, 2)
print(f'{rec_loc.shape=}')


nt = 1000
dt = 0.0004
freq = 25.0
peak_time = 1.5 / freq
dy, dx = 4.0, 4.0
v = torch.ones(ny, nx).to(device) * 1500.0
v = v[:ny, :nx]
nonhomo = False
if nonhomo:
    v[ny // 2, :] = 3000.0
    v[:, 3 * nx // 4] = 4500.0
    v[3 * ny // 4, :] = 2000.0
    v[:, 3 * ny // 4] = 4000.0
    beta = [4.0, 4.0]
halfwidth = [70, 70]
beta = [4.0, 4.0]

t = torch.linspace(0, nt * dt, nt, device=device)
source_amplitudes_true = (
    dw.wavelets.ricker(freq, nt, dt, peak_time).repeat(n_shots, 1).to(device)
)
source_amplitudes = SourceAmplitudes(
    ny=ny,
    nx=nx,
    # init_loc0=100.5,
    # init_loc1=60.6753435,
    init_loc0=60.0,
    init_loc1=64.9,
    source_trace=source_amplitudes_true,
    beta=beta[0],
    halfwidth=halfwidth[0],
)

ref_amplitudes = SourceAmplitudes(
    ny=ny,
    nx=nx,
    init_loc0=60.0,
    init_loc1=65.0,
    source_trace=source_amplitudes_true,
    beta=beta[0],
    halfwidth=halfwidth[0],
    trainable=False,
)

final_src_loc = source_locations_all.view(n_shots, -1, 2)
pml_width = 20


def forward(*, amps, msg=''):
    # check_nans(amps, name='amps', msg=msg)
    u = dw.scalar(
        v,
        [dy, dx],
        dt=dt,
        source_amplitudes=amps,
        source_locations=final_src_loc,
        receiver_locations=rec_loc,
        pml_width=pml_width,
    )[-1]

    # throw error if any NaNs are detected
    # check_nans(u, name='output', msg=msg)

    return u


obs_data_true = forward(amps=ref_amplitudes(), msg='True')

n_epochs = 100
optimizer = torch.optim.SGD(source_amplitudes.parameters(), lr=0.01)
loss = torch.nn.MSELoss()

num_frames = 10
save_freq = max(1, n_epochs // num_frames)
# param_history = [source_amplitudes().detach().clone()
param_history = []
# loss_history = [
#     loss(
#         obs_data_true, forward(amps=source_amplitudes(), msg='Initial').detach()
#     )
# ]
loss_history = []
grad_norm_history = []
for epoch in range(n_epochs):
    # print(f'loc: [{source_amplitudes.loc0.item()}, {source_amplitudes.loc1.item()}]')
    loc = source_amplitudes.loc()
    # print(f'{loc=}')
    optimizer.zero_grad()
    obs_data = forward(amps=source_amplitudes(), msg=f'{epoch=}')
    loss_val = loss(obs_data, obs_data_true)
    loss_val.backward()
    if epoch % save_freq == 0:
        param_history.append(source_amplitudes())
        loss_history.append(loss_val)

    optimizer.step()
    input(
        f'Epoch {epoch}, Loss: {loss_val.item()}, neg_loc_grad:'
        f' {list(-source_amplitudes.loc.get_grad().cpu().numpy())}'
    )
