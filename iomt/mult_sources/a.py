import deepwave as dw
import matplotlib.pyplot as plt
import torch
from misfit_toys.utils import bool_slice, clean_idx
from mh.typlotlib import save_frames, get_frames_bool

device = 'cuda:1'

ny, dy = 100, 4.0
nx, dx = 100, 4.0
nt, dt = 1000, 0.0001

v = torch.ones(ny, nx, device=device) * 1500.0

freq = 50.0
peak_time = 1.5 / freq

n_shots = 1

src_ny = ny - 2
src_nx = nx - 2
src_per_shot = src_ny * src_nx

# densely populate from 1 to src_ny - 1
ys = (
    torch.arange(1, src_ny + 1, device=device)
    .repeat_interleave(src_nx)
    .view(1, -1, 1)
)
xs = torch.arange(1, src_nx + 1, device=device).repeat(src_ny).view(1, -1, 1)

# Stack them together to form the source locations
source_locations = torch.cat((ys, xs), dim=2).expand(n_shots, -1, -1)

cy = ny // 2 * dy
cx = nx // 2 * dx

num_grid_sig = 5
mu = torch.tensor([cy, cx], device=device)
sigma = torch.tensor([dy * num_grid_sig, dx * num_grid_sig], device=device)

x = torch.arange(1, src_nx + 1, device=device) * dx
y = torch.arange(1, src_ny + 1, device=device) * dy

X, Y = torch.meshgrid(x, y)
Z = torch.exp(
    -(
        (X - mu[1]) ** 2 / (2 * sigma[1] ** 2)
        + (Y - mu[0]) ** 2 / (2 * sigma[0] ** 2)
    )
)

plt.imshow(Z.detach().cpu())
plt.savefig('a.jpg')
plt.colorbar()
plt.title('Gaussian Field')
plt.clf()


# time_signature = dw.wavelets.ricker(freq, nt, dt, peak_time)

time_signature = (
    dw.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, src_per_shot, 1)
    .to(device)
)
# time_signature = torch.rand(n_shots, src_per_shot, nt, device=device)
# input(time_signature.shape)
# input(Z.shape)
# input(Z.reshape(1, -1, 1).shape)

final_amps = Z.reshape(1, -1, 1) * time_signature
loop_amps = final_amps.squeeze().reshape(src_ny, src_nx, nt)


shape = (2, 2)
fig, axes = plt.subplots(*shape, figsize=(10, 10))
iter = bool_slice(*loop_amps.shape, none_dims=[0, 1], strides=[1, 1, 20])


def plotter(*, data, idx, fig, axes):
    print(clean_idx(idx))
    plt.clf()
    plt.subplot(*shape, 1)
    # plt.imshow(data[idx].detach().cpu(), aspect='auto', cmap='seismic')
    plt.imshow(
        data[idx].detach().cpu(),
        vmin=data.min(),
        vmax=data.max(),
        aspect='auto',
        cmap='seismic',
    )
    plt.title(f'Time={idx[-1]}')
    plt.colorbar()

    plt.subplot(*shape, 2)
    plt.imshow(Z.detach().cpu(), cmap='seismic', aspect='auto')
    plt.title('Gaussian Field')
    plt.colorbar()

    plt.subplot(*shape, 3)
    plt.plot(time_signature[0, 0].detach().cpu())
    plt.title('Representative Time Signature')
    plt.tight_layout()


frames = get_frames_bool(
    data=loop_amps,
    iter=iter,
    fig=fig,
    axes=axes,
    plotter=plotter,
)

save_frames(frames, path='final.gif')
