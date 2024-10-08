import os
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
from mh.core import hydra_out, DotDict, set_print_options, torch_stats
from misfit_toys.swiffer import dupe
from time import time, sleep
from deepwave.common import vpvsrho_to_lambmubuoyancy as get_lame

set_print_options(callback=torch_stats('all'))


def check_nans(u: torch.Tensor, *, name: str = 'output', msg: str = '') -> None:
    return u.nan_to_num(nan=0.0)


def report_tensor_status(d: DotDict, **kw):
    for k, v in d.items():
        # print(f'{k=}, {v.shape=}, {v.dtype=}, {v.device=}, {v.requires_grad=}', **kw)
        print(f'{k}={v}')


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
    c = DotDict({**c, 'rt': {'data': {}}})
    c.rt.dtype = torch.float32

    def get_path(x='', ext='pt'):
        return os.path.join(c.path, x.replace(f'.{ext}', '') + f'.{ext}')

    with open(get_path('metadata', 'pydict')) as f:
        metadata = DotDict(eval(f.read()))

    # meta_priority = {'ny', 'nx', 'nt', 'dy', 'dx', 'dt', 'freq', 'peak_time'}
    # meta_priority = {'ny', 'nx', 'nt', 'dy', 'dx', 'dt'}
    meta_priority = {'ny', 'nx', 'dy', 'dx'}

    keys = set(metadata.keys())

    assert meta_priority.issubset(keys), f'missing keys: {meta_priority - keys}'
    for k, v in metadata.items():
        if k in c.keys() and k in meta_priority:
            raise ValueError(
                f'KEY CLASH: {k} in hydra config and metadata...exiting'
            )
        if k in meta_priority:
            c[k] = v

    for e in os.listdir(c.path):
        if e.endswith('.pt'):
            c.rt.data[e.replace('.pt', '')] = torch.load(get_path(e)).to(
                c.device
            )

    c.peak_time = c.peak_time_factor / c.freq
    c.plt.final.save.path = hydra_out(c.plt.final.save.path)
    c.sleep_time = c.get('sleep_time', None)
    c.tol = c.get('tol', 1e-6)
    return c


@hydra.main(
    config_path='cfg/landscape/elastic', config_name='cfg', version_base=None
)
def main(cfg: DictConfig):
    with open(hydra_out('git_info.txt'), 'w') as f:
        f.write(git_dump_info())

    c = preprocess_cfg(cfg)

    print(c.pretty_str(max_length=30))

    # report_tensor_status(c.rt.data)
    # input()

    # c.rt.data.vp_true = c.eps + c.rt.data.vp_true
    # c.rt.data.vs_true = c.eps + c.rt.data.vs_true

    # convert km/s to m/s
    conversion_factor = 1e4
    c.rt.data.vp_true = conversion_factor * c.rt.data.vp_true
    c.rt.data.vs_true = conversion_factor * c.rt.data.vs_true
    c.rt.data.rho_true = conversion_factor * c.rt.data.rho_true

    zero_idx = c.rt.data.vs_true == 0.0
    scaling = 1.0 / 3.0
    c.rt.data.vs_true[zero_idx] = c.rt.data.vp_true[zero_idx] * scaling

    assert c.rt.data.vp_true.min() > 0.0
    assert c.rt.data.vs_true.min() > 0.0
    assert c.rt.data.rho_true.min() > 0.0
    assert (c.rt.data.vp_true >= c.rt.data.vs_true).all()

    mid_shot = c.n_shots // 2
    for k, v in c.rt.data.items():
        if not any(e in k for e in ['vp', 'vs', 'rho']):
            c.rt.data[k] = v[mid_shot].unsqueeze(0)

    # report_tensor_status(c.rt.data)

    # input(c.device)
    def forward(y_idx, x_idx):
        loc = torch.tensor([y_idx, x_idx], device=c.device)
        loc = loc.repeat(c.n_shots, 1).view(c.n_shots, -1, 2)
        res = dw.elastic(
            *get_lame(c.rt.data.vp_true, c.rt.data.vs_true, c.rt.data.rho_true),
            c.dy,
            dt=c.dt,
            source_amplitudes_y=c.rt.data.src_amp_y,
            source_locations_y=loc,
            receiver_locations_y=c.rt.data.rec_loc_y,
            source_amplitudes_x=c.rt.data.src_amp_x,
            source_locations_x=loc,
            receiver_locations_x=c.rt.data.rec_loc_x,
            accuracy=4,
            pml_freq=c.freq,
            pml_width=c.pml_width,
        )
        return torch.stack(res[-2:], dim=-1)

    # y_ref = src_loc[:, :, 0].item()
    # x_ref = src_loc[:, :, 1].item()
    y_ref = c.ny // 2
    x_ref = c.nx // 2
    obs_data = forward(y_ref, x_ref)
    input(f'{obs_data.min()}, {obs_data.max()}, {c.dt=}, {c.nt=}, {c.nt * c.dt=}, {c.rt.data.vp_true.min()=}, {c.rt.data.vs_true.min()=}, {c.rt.data.rho_true.min()=}, {c.rt.data.vp_true.max()=}, {c.rt.data.vs_true.max()=}, {c.rt.data.rho_true.max()=}')
    if torch.norm(obs_data) < c.tol:
        raise ValueError(f'obs_data is essentially zero {torch.norm(obs_data)=}')

    def error(y_idx, x_idx):
        u = forward(y_idx, x_idx)
        return torch.nn.functional.mse_loss(u, obs_data).item()

    y_srcs = torch.arange(c.delta, c.ny - c.delta, c.step)
    x_srcs = torch.arange(c.delta, c.nx - c.delta, c.step)
    # input(f'{y_srcs.shape=}, {x_srcs.shape=}')
    y_srcs[y_srcs.shape[0] // 2] = y_ref
    x_srcs[x_srcs.shape[0] // 2] = x_ref
    src_loc_y = (
        torch.cartesian_prod(y_srcs, x_srcs)
        .unsqueeze(0)
        .permute(1, 0, 2)
        .to(c.device)
    )
    src_amp_y = c.rt.data.src_amp_y.repeat(src_loc_y.shape[0], 1, 1)
    rec_loc_y = c.rt.data.rec_loc_y.repeat(src_loc_y.shape[0], 1, 1)

    src_loc_x = src_loc_y.clone()
    src_amp_x = c.rt.data.src_amp_x.repeat(src_loc_y.shape[0], 1, 1)
    rec_loc_x = c.rt.data.rec_loc_x.repeat(src_loc_y.shape[0], 1, 1)

    # Batch size decided on for a 24GB RTX 3090
    #     adjust according to your GPU VRAM
    batch_size = c.get('batch_size', 60)

    if c.sleep_time is not None:
        print(
            f'{src_loc_y.shape[0]} forward solves to be executed.\nYou have'
            f' {c.sleep_time} seconds to press CTRL+C to cancel immediately.\n'
            'Otherwise, there will be a significant delay from CPU <-> GPU signal interruption sending.\n'
        )
        sleep(c.sleep_time)

    # use numpy split to get the index slices
    idxs = torch.arange(0, src_loc_y.shape[0], batch_size)
    if idxs[-1] != src_loc_y.shape[0] - 1:
        idxs = torch.cat((idxs, torch.tensor([src_loc_y.shape[0]])))
    idxs = idxs.long().numpy()
    slices = [slice(idxs[i], idxs[i + 1], 1) for i in range(idxs.shape[0] - 1)]
    errors = torch.zeros(src_loc_y.shape[0], dtype=torch.float32) * 100.0
    final_wavefield_y = torch.zeros(src_loc_y.shape[0], *c.rt.data.vp_true.shape)
    final_wavefield_x = torch.zeros(src_loc_y.shape[0], *c.rt.data.vp_true.shape)
    num_slices = len(slices)
    for i, s in enumerate(slices):
        start = time()
        print(f'({s.start}, {s.stop}) -> ', end='', flush=True)
        res = dw.elastic(
            *get_lame(c.rt.data.vp_true, c.rt.data.vs_true, c.rt.data.rho_true),
            c.dy,
            dt=c.dt,
            source_amplitudes_y=src_amp_y[s],
            source_locations_y=src_loc_y[s],
            receiver_locations_y=rec_loc_y[s],
            source_amplitudes_x=src_amp_x[s],
            source_locations_x=src_loc_x[s],
            receiver_locations_x=rec_loc_x[s],
            accuracy=4,
            pml_freq=c.freq,
            pml_width=c.pml_width,
        )
        u = torch.stack(res[-2:], dim=-1)
        percent_complete = 100 * (i + 1) / num_slices
        print(f'{time()-start=}s ::: {percent_complete}%', flush=True)
        # errors[s] = torch.nn.functional.mse_loss(u, obs_data.repeat(s.stop - s.start, 1, 1)).cpu()
        for i in torch.arange(s.start, s.stop):
            errors[i] = (
                torch.nn.functional.mse_loss(u[i - s.start], obs_data.squeeze())
                .detach()
                .cpu()
            )
            w = c.pml_width
            final_wavefield_y[i] = res[0][i - s.start, w:-w, w:-w].detach().cpu()
            final_wavefield_x[i] = res[1][i - s.start, w:-w, w:-w].detach().cpu()

    torch.save(errors, hydra_out('errors.pt'))

    errors = errors.view(y_srcs.shape[0], x_srcs.shape[0])
    opts = dict(aspect='auto', cmap='seismic')
    plt.clf()
    plt.imshow(
        errors.T,
        **opts,
        extent=[y_srcs.min(), y_srcs.max(), x_srcs.min(), x_srcs.max()],
    )
    plt.colorbar()
    plt.savefig(hydra_out('errors.jpg'))

    plt.clf()
    plt.imshow(c.rt.data.vp_true.detach().cpu().squeeze().T, **opts)
    plt.colorbar()
    plt.savefig(hydra_out('vp_true.jpg'))

    plt.clf()
    plt.imshow(c.rt.data.vs_true.detach().cpu().squeeze().T, **opts)
    plt.colorbar()
    plt.savefig(hydra_out('vs_true.jpg'))

    final = DotDict({'subplot': {'shape': (2, 1), 'kw': {'figsize': (10, 10)}}})

    pcfg = c.plt.final
    fig, axes = plt.subplots(*pcfg.subplot.shape, **pcfg.subplot.setup_kw)
    iter = bool_slice(*final_wavefield_y.shape, **pcfg.iter)

    def plotter(*, data, idx, fig, axes):
        curr_src_x = src_loc_x[0][idx[0]].cpu().tolist()
        curr_src_y = src_loc_y[0][idx[0]].cpu().tolist()
        print(f'{clean_idx(idx)} -> {curr_src_x=}, {curr_src_y=}')
        plt.clf()
        plt.subplot(*pcfg.subplot.shape, pcfg.subplot.order[0])
        plt.imshow(data.xcomp[idx].T, **pcfg.imshow, vmin=data.xcomp.min(), vmax=data.xcomp.max())
        plt.colorbar()
        plt.title(f'{pcfg.title} X component {curr_src_x=}, {curr_src_y=}')

        plt.subplot(*pcfg.subplot.shape, pcfg.subplot.order[1])
        plt.imshow(data.ycomp[idx].T, **pcfg.imshow, vmin=data.ycomp.min(), vmax=data.ycomp.max())  
        plt.title(f'{pcfg.title} Y component {curr_src_x=}, {curr_src_y=}')
        plt.colorbar()
        # plt.savefig(f'{hydra_out(pcfg.save)}_{clean_idx(idx)}.jpg')

    frames = get_frames_bool(
        data=DotDict({'xcomp': final_wavefield_x, 'ycomp': final_wavefield_y}),
        iter=iter,
        plotter=plotter,
        fig=fig,
        axes=axes,
    )
    save_frames(frames, **c.plt.final.save)

    print(f"See {hydra_out('errors.jpg')}")
    print(f"See {hydra_out('vp_true.jpg')}")
    print(f"See {hydra_out('vs_true.jpg')}")

    with open('.latest', 'w') as f:
        f.write(f'cd {hydra_out()}')

    print('Run below to see the results\n    . .latest')


if __name__ == "__main__":
    main()
