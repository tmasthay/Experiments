import traceback
from typing import List
from matplotlib import patches, pyplot as plt
import torch
import deepwave as dw
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame
from mh.core import DotDict
from os.path import join as pj
from mh.typlotlib import (  # noqa: F401
    get_frames_bool,
    save_frames,
    bool_slice,
    clean_idx,  # noqa: F401
)  # noqa: F401
from time import time
import torch.nn.functional as F
from misfit_toys.utils import tslice
from returns.curry import curry


def frames_to_strides(*shape, none_dims=None, max_frames):
    none_dims = none_dims or []
    assert len(none_dims) == len(
        set(none_dims)
    ), f'Duplicates found in {none_dims=}'
    for i, dim in enumerate(none_dims):
        if dim < 0:
            none_dims[i] = len(shape) + dim
    n_active_dims = len(shape) - len(none_dims)
    frames_per_dim = int(max_frames ** (1 / n_active_dims))
    strides = [1 for e in shape]
    if frames_per_dim <= 1:
        return strides
    for i, dim in enumerate(shape):
        if i not in none_dims:
            strides[i] = dim // frames_per_dim
    return strides


def easy_imshow(
    data,
    *,
    transpose=False,
    imshow=None,
    colorbar=True,
    xlabel='',
    ylabel='',
    title='',
    extent=None,
    bound_data=None,
    clip=0.0,
    path=None,
    **kw,
):
    # input(xlabel)
    # input(ylabel)
    # input(extent)
    imshow = imshow or {}
    if transpose:
        data = data.T
    if extent is not None:
        imshow['extent'] = extent

    if bound_data is None:
        bound_data = data
    vmin, vmax = bound_data.min(), bound_data.max()
    imshow['vmin'] = vmin + clip * abs(vmin)
    imshow['vmax'] = vmax - clip * abs(vmax)

    plt.imshow(data.detach().cpu(), **imshow, **kw)
    if colorbar:
        plt.colorbar()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title)

    if path is not None:
        plt.savefig(path)
        print(f'Saved to {path}')


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


def get_grid_limits(*, sy, ny, dy, sx, nx, dx):
    max_y = sy + ny * dy
    max_x = sx + nx * dx
    return [sy, max_y, max_x, sx]
    # return [sy, sx, max_y, max_x]


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
    min_vs: float
):
    # .707 approx 1/sqrt(2)
    assert 0.0 < rel_vp_scaling <= 0.707
    assert 0.0 < global_scaling
    vs = global_scaling * torch.load(path, map_location=device)
    if vs.shape != vp.shape:
        vs = (
            F.interpolate(
                vs[None, None, ...],
                size=vp.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )

    zero_idx = vs == 0.0
    vs[zero_idx] = vp[zero_idx] * rel_vp_scaling
    
    true_min = min(0.707 * vp.min(), min_vs)
    vs = torch.clamp(vs, min=true_min)
    # vs = torch.clamp(vs, min=min(torch.sqrt(torch.tensor(2.0)) * vp.min(), min_vs))
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


# var1_ref * var2_ref = var1_actual * var2_actual = const
# -> return var2_actual = const / var1_actual
def const_prod_rescale(*, var1_ref: float, var2_ref: float, var1_actual: float):
    const_prod = var1_ref * var2_ref
    return const_prod / var1_actual


def int_scale(*, ref: int, scale: float = 1.0):
    return int(ref * scale)


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


class MyL2Loss(torch.nn.Module):
    def __init__(self, ref_data):
        super().__init__()
        self.ref_data = ref_data

    def forward(self, data):
        tmp = self.ref_data.repeat(data.shape[0], 1, 1, 1)
        assert (
            tmp.shape == data.shape
        ), f'{self.ref_data.shape=}, {tmp.shape=}, {data.shape=}'
        diff = tmp - data
        v = torch.norm(diff, dim=-1).norm(dim=-1).norm(dim=-1)
        assert v.shape[0] == data.shape[0], f'{v.shape=}, {data.shape=}'
        return v


def elastic_landscape_loop(c):
    def forward(s):
        # vp,vs,rho technically
        #     should be moved only once
        #     but not worth refactoring right now
        # would just require changing device: ${device} to device: ${gpu}
        # in the config file
        vp = c.rt.vp.to(c.gpu)
        vs = c.rt.vs.to(c.gpu)
        rho = c.rt.rho.to(c.gpu)
        src_amp_y = c.rt.src_amp.y[s].to(c.gpu)
        src_amp_x = c.rt.src_amp.x[s].to(c.gpu)
        src_loc_y = c.rt.src_loc.y[s].to(c.gpu)
        src_loc_x = c.rt.src_loc.x[s].to(c.gpu)
        rec_loc_y = c.rt.rec_loc.y[s].to(c.gpu)
        rec_loc_x = c.rt.rec_loc.x[s].to(c.gpu)
        u = easy_elastic(
            vp=vp,
            vs=vs,
            rho=rho,
            grid_spacing=[c.grid.dy, c.grid.dx],
            dt=c.grid.dt,
            source_amplitudes_y=src_amp_y,
            source_locations_y=src_loc_y,
            source_amplitudes_x=src_amp_x,
            source_locations_x=src_loc_x,
            receiver_locations_x=rec_loc_x,
            receiver_locations_y=rec_loc_y,
            **c.solver,
        )
        wavefield = torch.stack(u[:2], dim=-1)
        w = c.solver.get('pml_width', 20)
        wavefield = wavefield[:, w:-w, w:-w, :].to(c.device)
        final_obs = torch.stack(u[-2:], dim=-1).to(c.device)
        return wavefield, final_obs

    idxs = torch.arange(0, c.rt.src_loc.y.shape[0], c.batch_size)
    if idxs[-1] != c.rt.src_loc.y.shape[0]:
        idxs = torch.cat([idxs, torch.tensor([c.rt.src_loc.y.shape[0]])])
    slices = [slice(idxs[i], idxs[i + 1]) for i in range(idxs.shape[0] - 1)]

    errors = torch.rand(c.src.n_horz * c.src.n_deep, device=c.device) * 100.0
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
        # print(msg, flush=True, end='\r')
        print(msg, flush=True)

    start_time = time()

    # ref_data = ref_data.permute(0, 3, 1, 2)
    my_loss = c.rt.loss.constructor(
        ref_data, *c.rt.loss.get('args', []), **c.rt.loss.get('kw', {})
    )
    # c = runtime_reduce(c, )
    for i, s in enumerate(slices):
        report_progress(i)
        final_wavefields[s], obs[s] = forward(s)
        v = my_loss(obs[s])
        # raise ValueError(f'{v.shape=}, {errors[s].shape=}')
        errors[s] = my_loss(obs[s]).view(*errors[s].shape)
    assert errors.min() <= 1e-8, f'{errors.min()=}'
    total_forward_solve_time = time() - start_time
    avg_forward_solve_time = total_forward_solve_time / c.rt.src_loc.y.shape[0]
    print(
        f'Total solve time: {total_forward_solve_time:.2f}s\n    avg:'
        f' {avg_forward_solve_time:.2f}s'
    )

    errors = errors.view(c.src.n_horz, c.src.n_deep)
    final_wavefields = final_wavefields.view(
        c.src.n_horz, c.src.n_deep, *c.rt.vp.shape, 2
    )
    return DotDict(
        {'final_wavefields': final_wavefields, 'obs': obs, 'errors': errors}
    )


class EasyW1Loss(torch.nn.Module):
    def __init__(self, ref_data, *, renorm=None, dim=-1, eps=1e-8):
        super().__init__()
        if renorm is None:
            renorm = torch.abs

        self.renorm = renorm
        self.dim = dim
        self.eps = eps
        self.cdf = self.__cdf__(ref_data, renorm=renorm, eps=eps, dim=dim)
        # input(f'{self.cdf.shape=}')

    @staticmethod
    def __cdf__(data, *, renorm, dim=-1, eps=1e-8):
        pdf = renorm(data)
        assert pdf.min() >= 0.0, f'{pdf.min()=}, should be >= 0.0'

        v = torch.cumulative_trapezoid(pdf, dim=dim)
        assert torch.isnan(v).sum() == 0, f'{v=}'
        divider = tslice(v, dims=[dim]).unsqueeze(dim)
        assert divider.min() > 0.0, f'{divider.min()=}, should be > 0.0'
        u = v / (eps + tslice(v, dims=[dim]).unsqueeze(dim=dim))
        assert u.max() <= 1.0, f'{u.max().item()=}, should be <= 1.0'
        assert u.min() >= 0.0, f'{u.min().item()=}, should be >= 0.0'
        return u

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        lcl_cdf = EasyW1Loss.__cdf__(data, renorm=self.renorm, dim=self.dim, eps=self.eps)
        # raise RuntimeError(f'{self.cdf.shape=}, {lcl_cdf.shape=}')
        # diff = (lcl_cdf - self.cdf).abs()
        # return torch.sum(diff**2, dim=self.dim)
        diff = lcl_cdf - self.cdf
        integrand = diff**2
        res = torch.mean(integrand, dim=self.dim)
        
        # consider refactoring later if you want something
        # more general, but for now this is fine
        res_flat = res.view(res.shape[0], -1)
        return res_flat.mean(dim=1)
        # raise RuntimeError(f'{torch.mean(integrand, dim=self.dim).shape=}') 


def rel_label(*, label, diff, num, unit):
    max_val = diff * num
    return f'{label} ({max_val:.2e} {unit})'


def rel2abs(*, rel_coords, diff, start=0.0):
    _min, _max = rel_coords
    true_min = start + _min * diff
    true_max = start + _max * diff
    return [true_min, true_max]


def rel2abs_extent(*, lower_left, upper_right, ny, nx, dy, dx, sy=0.0, sx=0.0):
    # we assume here that y = "horizontal" and x = "vertical"
    # This is just to remain consistent with the rest of the code
    # despite it being nonsensical notation.
    # Furthermore, we assume POSITIVE depth cooridnate points DOWNWARDS
    # Hence just think about this and it's not that bad, even if the formulae
    # below look weird.

    # for clarity we write
    #     y <--> horz
    #    x <--> depth

    min_horz = sy + lower_left[0] * dy * ny
    max_horz = sy + upper_right[0] * dy * ny

    min_depth = sx + upper_right[1] * dx * nx
    max_depth = sx + lower_left[1] * dx * nx

    # input(f'{lower_left=}, {upper_right=}, {ny=}, {nx=}, {dy=}, {dx=}, {sy=}, {sx=}')
    # input(f'    ----> {min_horz=}, {max_horz=}, {min_depth=}, {max_depth=}')

    # res = [min_horz, max_horz, min_depth, max_depth]
    # input(f'{res=}')
    return [min_horz, max_horz, min_depth, max_depth]


def add_box(coords, **kw):
    """
    Adds a rectangular box to the current plot.

    Parameters:
        coords: List or array of the form [xmin, xmax, ymin, ymax].
        color: Color of the box's edge.
        linewidth: Thickness of the box's edge.
    """
    xmin, xmax, ymin, ymax = coords
    # Calculate the width and height
    width = xmax - xmin
    height = ymax - ymin

    # kw['facecolor'] = 'none'
    # Create the rectangle
    rect = patches.Rectangle((xmin, ymin), width, height, **kw)
    # Add the rectangle to the current axis
    ax = plt.gca()
    ax.add_patch(rect)


def abs_label(*, label, unit):
    return f'{label} ({unit})'


def merge_dot_dict(a: DotDict, b: DotDict) -> DotDict:
    return DotDict({**a.dict(), **b.dict()})


def dump_tensors(c: DotDict, *, path):
    rt_errors_list = []

    def add_err(e, msg):
        stars = 80 * '*'
        rt_errors_list.append(stars)
        rt_errors_list.append(msg)
        rt_errors_list.append(e)
        rt_errors_list.append(stars)
        rt_errors_list.append('\n\n')

    def extend_name(name, k):
        return f'{name}___{k}' if name else k

    verbose = c.get('verbose', True)
    q = [(c, '')]

    while q:
        curr, name = q.pop()
        for k, v in curr.items():
            if isinstance(v, DotDict):
                q.append((v, extend_name(name, k)))
            elif isinstance(v, torch.Tensor):
                full_path = pj(path, extend_name(name, k) + '.pt')
                try:
                    torch.save(v.detach().cpu(), full_path)
                except:
                    add_err(
                        f'Error saving {k} to {full_path}',
                        traceback.format_exc(),
                    )
                if verbose:
                    print(f'Saved {k} to {full_path}')
            elif isinstance(v, list) or isinstance(v, tuple):
                for i, e in enumerate(v):
                    if isinstance(e, DotDict):
                        q.append((e, extend_name(name, f'{k}_{i}')))
                    elif isinstance(e, torch.Tensor):
                        full_path = pj(
                            path, extend_name(name, f'{k}_{i}') + '.pt'
                        )
                        torch.save(e.detach().cpu(), full_path)
                        if verbose:
                            print(f'Saved {k}_{i} to {full_path}')

    return '\n'.join(rt_errors_list)


def dump_and_plot_tensors(c: DotDict, *, path):
    assert 'plt' in c.postprocess

    tensor_errors = dump_tensors(c, path=path)
    plot_errors = plot_landscape(c, path=path)

    if tensor_errors and plot_errors:
        final_msg = (
            f'BOTH tensor AND plot errors:\n\n{tensor_errors}\n\n{plot_errors}'
        )
    elif tensor_errors:
        final_msg = f'Tensor errors:\n\n{tensor_errors}'
    elif plot_errors:
        final_msg = f'Plot errors:\n\n{plot_errors}'
    else:
        final_msg = ''

    if final_msg:
        raise RuntimeError(final_msg)


def plot_tensors(c: DotDict, *, path):
    assert 'plt' in c.postprocess

    plot_errors = plot_landscape(c, path=path)

    if plot_errors:
        raise RuntimeError(f'Plot errors:\n\n{plot_errors}')


def plot_landscape(c: DotDict, *, path):
    assert 'rt' in c
    assert 'res' in c.rt

    wavefields = c.rt.res.final_wavefields
    obs = c.rt.res.obs
    errors = c.rt.res.errors

    opts = c.postprocess.plt

    src_loc_y = (
        c.rt.src_loc.y.detach().cpu().view(c.src.n_horz, c.src.n_deep, 2)
    )
    src_loc_x = (
        c.rt.src_loc.x.detach().cpu().view(c.src.n_horz, c.src.n_deep, 2)
    )
    # errors_flat = errors.view(-1)

    def plot_errors():
        plt.clf()
        easy_imshow(
            errors.cpu(),
            path=pj(path, opts.errors.other.filename),
            **opts.errors.filter(exclude=['other']),
        )
        plt.clf()

    def plot_medium():
        plt.clf()

        def toggle_subplot(i):
            plt.subplot(*subp_med.shape, subp_med.order[i - 1])

        subp_med = opts.medium.subplot
        fig, axes = plt.subplots(*subp_med.shape, **subp_med.kw)
        plt.suptitle(subp_med.suptitle)

        toggle_subplot(1)
        easy_imshow(c.rt.vp.cpu().T, **opts.medium.vp.imshow)

        toggle_subplot(2)
        easy_imshow(c.rt.vs.cpu().T, **opts.medium.vs.imshow)

        toggle_subplot(3)
        easy_imshow(c.rt.rho.cpu().T, **opts.medium.rho.imshow)

        filename = f'{pj(path, opts.medium.filename)}.png'
        plt.savefig(filename)
        print(f'Saved vp,vs,rho to {filename}')
        plt.clf()

    def plot_wavefields():
        plt.clf()

        def plotter(*, data, idx, fig, axes):
            yx, yy = src_loc_y[idx[1], idx[0]].tolist()
            xx, xy = src_loc_x[idx[1], idx[0]].tolist()

            # yx, yy = src_loc_y[idx[1], idx[0]].tolist()
            # xx, xy = src_loc_x[idx[1], idx[0]].tolist()

            yx = c.grid.dx * yx
            xx = c.grid.dx * xx

            yy = c.grid.dy * yy
            xy = c.grid.dy * xy

            # input(f'{yx=}, {yy=}')
            # input(f'{xx=}, {xy=}')

            # input(f'{yx=}, {yy=}, {xx=}, {xy=}')
            # input([yx, yy, xx, xy])

            # if 'extent' in opts.wavefields.y:
            #     yy *= c.grid.dy
            #     xy *= c.grid.dy

            #     xx *= c.grid.dx
            #     yx *= c.grid.dx

            # input([yy, xy, yx, xx])

            subp_wave = opts.wavefields.subplot
            if 'other' in opts.wavefields.y and opts.wavefields.y.other.get(
                'static', False
            ):
                opts.wavefields.y.bound_data = data[..., 0]
                opts.wavefields.x.bound_data = data[..., 1]

            plt.clf()
            plt.subplot(*subp_wave.shape, subp_wave.order[0])
            plt.scatter([yy], [yx], **opts.wavefields.y.other.marker)
            easy_imshow(
                data[idx][..., 0].cpu().T, **opts.wavefields.y.filter(['other'])
            )
            add_box(
                c.postprocess.plt.errors.extent,
                **c.postprocess.plt.wavefields.y.other.box,
            )

            plt.subplot(*subp_wave.shape, subp_wave.order[1])
            plt.scatter([xy], [xx], **opts.wavefields.x.other.marker)
            easy_imshow(
                data[idx][..., 1].cpu().T, **opts.wavefields.x.filter(['other'])
            )
            add_box(
                c.postprocess.plt.errors.extent,
                **c.postprocess.plt.wavefields.x.other.box,
            )

        subp_wave = opts.wavefields.subplot
        fopts_wave = opts.wavefields.frames
        fig, axes = plt.subplots(*subp_wave.shape, **subp_wave.kw)

        strides = frames_to_strides(
            *wavefields.shape,
            none_dims=fopts_wave.iter.none_dims,
            max_frames=fopts_wave.max_frames,
        )
        iter_wave = bool_slice(
            *wavefields.shape, **fopts_wave.iter, strides=strides
        )
        frames = get_frames_bool(
            data=wavefields, iter=iter_wave, plotter=plotter, fig=fig, axes=axes
        )
        filename = pj(path, opts.wavefields.filename)
        save_frames(frames, path=filename)
        print(f'\nSaved wavefields to {pj(path, f"{filename}.gif")}\n')
        plt.clf()

    def plot_obs():
        plt.clf()

        def plotter_obs(*, data, idx, fig, axes):
            subp_obs = opts.obs.subplot
            if 'other' in opts.obs.y and opts.obs.y.other.get('static', False):
                opts.obs.y.bound_data = data[..., 0]
                opts.obs.x.bound_data = data[..., 1]

            plt.clf()
            plt.subplot(*subp_obs.shape, subp_obs.order[0])
            easy_imshow(
                data[idx][..., 0].cpu().T, **opts.obs.y.filter(['other'])
            )

            plt.subplot(*subp_obs.shape, subp_obs.order[1])
            easy_imshow(
                data[idx][..., 1].cpu().T, **opts.obs.x.filter(['other'])
            )

        subp_obs = opts.obs.subplot
        fopts_obs = opts.obs.frames
        fig, axes = plt.subplots(*subp_obs.shape, **subp_obs.kw)
        strides = frames_to_strides(
            *obs.shape,
            none_dims=fopts_obs.iter.none_dims,
            max_frames=fopts_obs.max_frames,
        )
        iter_obs = bool_slice(*obs.shape, **fopts_obs.iter, strides=strides)
        frames = get_frames_bool(
            data=obs, iter=iter_obs, plotter=plotter_obs, fig=fig, axes=axes
        )
        filename_obs = pj(path, opts.obs.filename)
        save_frames(frames, path=filename_obs)
        print(f'\nSaved obs to {pj(path, f"{filename_obs}.gif")}\n')
        plt.clf()

    rt_error_list = []

    def add_err(e, msg):
        stars = 80 * '*'
        rt_error_list.append(stars)
        rt_error_list.append(msg)
        rt_error_list.append(e)
        rt_error_list.append(stars)
        rt_error_list.append('\n\n')

    try:
        plot_errors()
    except:
        add_err('Error plotting errors', traceback.format_exc())

    try:
        plot_medium()
    except:
        add_err('Error plotting medium', traceback.format_exc())

    try:
        plot_wavefields()
    except:
        add_err('Error plotting wavefields', traceback.format_exc())

    try:
        plot_obs()
    except:
        add_err('Error plotting obs', traceback.format_exc())

    return '\n'.join(rt_error_list)
