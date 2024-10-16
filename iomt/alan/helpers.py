from typing import List
from matplotlib import pyplot as plt
import torch
import deepwave as dw
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from mh.core import DotDict
from os.path import join as pj
from mh.typlotlib import get_frames_bool, save_frames, bool_slice, clean_idx
from time import time
import torch.nn.functional as F


def fixed_depth_rec(
    *,
    n_shots: int,
    rec_per_shot: int,
    first: int,
    drec: int,
    depth: int,
    device: str = 'cpu',
):
    builder = torch.zeros(n_shots, rec_per_shot, 2, dype=int)
    sweep = torch.arange(first, first + rec_per_shot * drec, drec)
    builder[..., 0] = sweep.repeat(n_shots, 1)
    builder[..., 1] = depth
    return builder.to(device)


def rel_fixed_depth_rec(
    *,
    n_shots: int,
    rec_per_shot: int,
    first: float,
    last: float,
    ny: int,
    depth: int,
    device: str = 'cpu',
):
    assert (
        0 <= first <= last <= 1
    ), f'{first=}, {last=}, expected 0 <= first <= last <= 1'

    first_idx = first * ny
    last_idx = last * ny
    sweep = torch.linspace(first_idx, last_idx, rec_per_shot).int()

    sweep_list = sweep.tolist()
    assert len(set(sweep_list)) == len(sweep_list), (
        f'{sweep_list=}, expected no duplicates, check {first=}, {last=},'
        f' {ny=}, {rec_per_shot=} leads to {first_idx=}, {last_idx=}'
    )
    builder = torch.zeros(n_shots, rec_per_shot, 2, dtype=int)
    builder[..., 0] = sweep.repeat(n_shots, 1)
    builder[..., 1] = depth
    return builder.to(device)


def landscape_sources(
    *,
    lower_left: List[float],
    upper_right: List[float],
    ny: int,
    nx: int,
    n_srcs_horz: int,
    n_srcs_deep: int,
    device: str,
):
    assert 0 <= lower_left[0] <= upper_right[0] <= 1
    assert 0 <= upper_right[1] <= lower_left[1] <= 1

    leftmost = lower_left[0] * ny
    rightmost = upper_right[0] * ny
    shallowest = upper_right[1] * nx
    deepest = lower_left[1] * nx

    horz_grid = torch.linspace(leftmost, rightmost, n_srcs_horz).int()
    deep_grid = torch.linspace(shallowest, deepest, n_srcs_deep).int()

    if len(set(horz_grid.tolist())) != len(horz_grid.tolist()):
        raise ValueError('Duplicate horizontal sources')
    if len(set(deep_grid.tolist())) != len(deep_grid.tolist()):
        raise ValueError('Duplicate deep sources')

    grid = torch.cartesian_prod(horz_grid, deep_grid)
    return grid[:, None, :].to(device)


def ricker_sources(
    *,
    n_srcs: float,
    freq: float,
    time_peak_factor: float,
    device: str = 'cpu',
    nt: int,
    dt: float,
    scale: float,
):
    t = torch.linspace(0.0, nt * dt, nt).to(device)
    peak = time_peak_factor / freq
    t = t - peak
    u = (
        scale
        * (1 - 2 * (torch.pi * freq * t) ** 2)
        * torch.exp(-((torch.pi * freq * t) ** 2))
    )
    return u[None, None, :].repeat(n_srcs, 1, 1)


def validate_elastic_loop_assertions(c):
    assert 'rt' in c
    assert 'src_loc' in c.rt
    assert 'rec_loc' in c.rt
    assert 'src_amp' in c.rt
    assert 'vp' in c.rt
    assert 'vs' in c.rt
    assert 'rho' in c.rt
    assert 'y' in c.rt.src_loc
    assert 'x' in c.rt.src_loc
    assert 'y' in c.rt.rec_loc
    assert 'x' in c.rt.rec_loc
    assert 'y' in c.rt.src_amp
    assert 'x' in c.rt.src_amp

    assert c.rt.src_loc.y.shape == c.rt.src_loc.x.shape
    assert c.rt.rec_loc.y.shape == c.rt.rec_loc.x.shape
    assert c.rt.rec_loc.y.shape[0] == c.rt.src_loc.y.shape[0]
    assert c.rt.src_amp.y.shape[0] == c.rt.src_loc.y.shape[0]

    def check_device(x, k):
        assert x.device == torch.device(
            c.device
        ), f'{k} on {x.device}, not {c.device}'

    check_device(c.rt.src_loc.y, 'src_loc.y')
    check_device(c.rt.src_loc.x, 'src_loc.x')
    check_device(c.rt.rec_loc.y, 'rec_loc.y')
    check_device(c.rt.rec_loc.x, 'rec_loc.x')
    check_device(c.rt.src_amp.y, 'src_amp.y')
    check_device(c.rt.src_amp.x, 'src_amp.x')
    check_device(c.rt.vp, 'vp')
    check_device(c.rt.vs, 'vs')
    check_device(c.rt.rho, 'rho')


def load_clamp_vs(
    *,
    path: str,
    device: str,
    vp: torch.Tensor,
    rel_vp_scaling: float,
    global_scaling: float,
):
    assert 0.0 < rel_vp_scaling <= 1.0
    assert 0.0 < global_scaling
    vs = global_scaling * torch.load(path, map_location=device)
    if vs.shape != vp.shape:
        vs = F.interpolate(
            vs[None, None, ...],
            size=vp.shape[-2:],
            mode='bilinear',
            align_corners=True,
        ).squeeze(0).squeeze(0)
        
    zero_idx = vs == 0.0
    vs[zero_idx] = vp[zero_idx] * rel_vp_scaling
    return vs


def load_scale_resample(
    *, path: str, device: str, scaling: float = 1.0, ny: int, nx: int
):
    u = torch.load(path, map_location=device) * scaling

    if ny != u.shape[0] or nx != u.shape[1]:
        u = F.interpolate(
            u[None, None, ...],
            size=(ny, nx),
            mode='bilinear',
            align_corners=True,
        )
        
    return u.squeeze(0).squeeze(0)


def easy_elastic(
    *,
    vp: torch.Tensor,
    vs: torch.Tensor,
    rho: torch.Tensor,
    grid_spacing: list[float],
    dt: float,
    source_amplitudes_y: torch.Tensor,
    source_locations_y: torch.Tensor,
    source_amplitudes_x: torch.Tensor,
    source_locations_x: torch.Tensor,
    receiver_locations_y: torch.Tensor,
    receiver_locations_x: torch.Tensor,
    **kw,
):
    return dw.elastic(
        *get_lame(vp=vp, vs=vs, rho=rho),
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        source_amplitudes_x=source_amplitudes_x,
        source_locations_x=source_locations_x,
        receiver_locations_x=receiver_locations_x,
        receiver_locations_y=receiver_locations_y,
        **kw,
    )


def elastic_landscape_loop(c):
    def forward(s):
        u = easy_elastic(
            vp=c.rt.vp,
            vs=c.rt.vs,
            rho=c.rt.rho,
            grid_spacing=[c.grid.dy, c.grid.dx],
            dt=c.grid.dt,
            source_amplitudes_y=c.rt.src_amp.y[s],
            source_locations_y=c.rt.src_loc.y[s],
            source_amplitudes_x=c.rt.src_amp.x[s],
            source_locations_x=c.rt.src_loc.x[s],
            receiver_locations_x=c.rt.rec_loc.x[s],
            receiver_locations_y=c.rt.rec_loc.y[s],
            **c.solver,
        )
        wavefield = torch.stack(u[:2], dim=-1)
        w = c.solver.get('pml_width', 20)
        wavefield = wavefield[:, w:-w, w:-w, :]
        return wavefield, torch.stack(u[-2:], dim=-1)

    def loss(a, b):
        tmp = a.repeat(b.shape[0], 1, 1, 1)
        assert tmp.shape == b.shape, f'{a.shape=}, {tmp.shape=}, {b.shape=}'
        diff = tmp - b
        v = torch.norm(diff, dim=-1).norm(dim=-1).norm(dim=-1)
        assert v.shape[0] == b.shape[0], f'{v.shape=}, {b.shape=}'
        return v

    idxs = torch.arange(0, c.rt.src_loc.y.shape[0], c.batch_size)
    if idxs[-1] != c.rt.src_loc.y.shape[0]:
        idxs = torch.cat([idxs, torch.tensor([c.rt.src_loc.y.shape[0]])])
    slices = [slice(idxs[i], idxs[i + 1]) for i in range(idxs.shape[0] - 1)]

    errors = torch.rand(c.src.n_horz, c.src.n_deep, device=c.device) * 100.0
    final_wavefields = torch.zeros(
        c.rt.src_loc.y.shape[0], *c.rt.vp.shape, 2, device=c.device
    )
    obs = torch.zeros(
        c.rt.src_loc.y.shape[0], c.rec.n_recs, c.grid.nt, 2, device=c.device
    )

    ref_idx = c.rt.src_loc.y.shape[0] // 2
    ref_wavefield, ref_data = forward(slice(ref_idx, ref_idx + 1, 1))

    num_slices = len(slices)
    start_time = time()

    def report_progress(i):
        msg = f'{i+1}/{num_slices}...'
        if i > 0:
            total_run_time = time() - start_time
            avg_run_time = total_run_time / i
            remaining_time = avg_run_time * (num_slices - i)
            sep = '    '
            msg += (
                f'{sep}AVG: {avg_run_time:.2f}s'
                f'{sep}TOTAL: {total_run_time:.2f}s'
                f'{sep}ETR: {remaining_time:.2f}s'
            )
        print(msg, flush=True, end='\r')

    for i, s in enumerate(slices):
        report_progress(i)
        final_wavefields[s], obs[s] = forward(s)
        errors[s] = loss(ref_data, obs[s]).view(*errors[s].shape)

    errors = errors.view(c.src.n_horz, c.src.n_deep)
    final_wavefields = final_wavefields.view(
        c.src.n_horz, c.src.n_deep, *c.rt.vp.shape, 2
    )
    return final_wavefields, obs, errors


def dump_tensors(c: DotDict, *, path):
    def extend_name(name, k):
        return f'{name}_{k}' if name else k

    verbose = c.get('verbose', True)
    q = [(c, '')]

    while q:
        curr, name = q.pop()
        for k, v in curr.items():
            if isinstance(v, DotDict):
                q.append((v, extend_name(name, k)))
            elif isinstance(v, torch.Tensor):
                full_path = pj(path, extend_name(name, k) + '.pt')
                torch.save(v.detach().cpu(), full_path)
                if verbose:
                    print(f'Saved {k} to {full_path}')


def dump_and_plot_tensors(c: DotDict, *, path):
    assert 'plt' in c.postprocess

    dump_tensors(c, path=path)
    plot_landscape(c, path=path)


def plot_landscape(c: DotDict, *, path):
    assert 'rt' in c
    assert 'res' in c.rt

    wavefields = c.rt.res[0]
    obs = c.rt.res[1]
    errors = c.rt.res[2]

    src_loc_y = (
        c.rt.src_loc.y.detach().cpu().view(c.src.n_horz, c.src.n_deep, 2)
    )
    src_loc_x = (
        c.rt.src_loc.x.detach().cpu().view(c.src.n_horz, c.src.n_deep, 2)
    )
    errors_flat = errors.view(-1)

    plt.imshow(errors.cpu(), aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.savefig(pj(path, 'landscape.png'))
    print(f'Saved landscape to {pj(path, "landscape.png")}')

    def plotter(*, data, idx, fig, axes):
        try:
            yx, yy = src_loc_y[idx[0], idx[1]].tolist()
            xx, xy = src_loc_x[idx[0], idx[1]].tolist()
        except Exception as e:
            tmp = {
                'idx': clean_idx(idx),
                'yshape': src_loc_y.shape,
                'xshape': src_loc_x.shape,
            }
            for k, v in tmp.items():
                print(f'{k}: {v}\n')
            raise e
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.scatter([yx], [yy], c='r', s=100, marker='*')
        plt.imshow(data[idx][..., 0].cpu().T, aspect='auto', cmap='seismic')
        plt.colorbar()
        plt.title(f'Y component of wavefield\n{clean_idx(idx[:2])}')

        plt.subplot(2, 1, 2)
        plt.scatter([xx], [xy], c='r', s=100, marker='*')
        plt.imshow(data[idx][..., 1].cpu().T, aspect='auto', cmap='seismic')
        plt.colorbar()
        plt.title(f'X component of wavefield\n{clean_idx(idx[:2])}')

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    iter = bool_slice(*wavefields.shape, none_dims=[-3, -2, -1])
    frames = get_frames_bool(
        data=wavefields, iter=iter, plotter=plotter, fig=fig, axes=axes
    )
    save_frames(frames, path=pj(path, 'wavefields'))
    print(f'\nSaved wavefields to {pj(path, "wavefields.gif")}\n')

    opts = {'cmap': 'seismic', 'aspect': 'auto'}
    static = True
    if static:
        opts1 = {
            **opts,
            **{'vmin': obs[..., 0].min(), 'vmax': obs[..., 0].max()},
        }
        opts2 = {
            **opts,
            **{'vmin': obs[..., 1].min(), 'vmax': obs[..., 1].max()},
        }
    else:
        opts1 = opts2 = opts

    def plotter_obs(*, data, idx, fig, axes):
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.imshow(data[idx][..., 0].cpu().T, **opts1)
        plt.colorbar()
        plt.title(
            'Y component of observed'
            f' data\n{clean_idx(idx[:2])}\n{errors_flat[idx[0]]=}'
        )

        plt.subplot(2, 1, 2)
        plt.imshow(data[idx][..., 1].cpu().T, **opts2)
        plt.colorbar()
        plt.title(
            'X component of observed'
            f' data\n{clean_idx(idx[:2])}\n{errors_flat[idx[0]]}'
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    iter = bool_slice(*obs.shape, none_dims=[-3, -2, -1])
    frames = get_frames_bool(
        data=obs, iter=iter, plotter=plotter_obs, fig=fig, axes=axes
    )
    save_frames(frames, path=pj(path, 'obs'))
    print(f'\nSaved obs_data to {pj(path, "obs.gif")}\n')
    
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plt.suptitle("Elastic seismic medium")
    
    plt.subplot(1, 3, 1)
    plt.imshow(c.rt.vp.cpu().T, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.title("Vp")

    plt.subplot(1, 3, 2)
    plt.imshow(c.rt.vs.cpu().T, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.title("Vs")
    
    plt.subplot(1, 3, 3)
    plt.imshow(c.rt.rho.cpu().T, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.title("Rho")
    
    plt.savefig(pj(path, 'medium.png'))
    print(f'Saved vp,vs,rho to {pj(path, "medium.png")}')
    
    
