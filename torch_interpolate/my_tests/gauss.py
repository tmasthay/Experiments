from matplotlib import pyplot as plt
from torch_interpolations import RegularGridInterpolator as RGI
import torch
import hydra
from omegaconf import DictConfig
from time import time
from mh.core import DotDict, set_print_options, torch_stats
from misfit_toys.utils import bool_slice
from mh.typlotlib import get_frames_bool, save_frames

set_print_options(callback=torch_stats('all'))


def gauss(*, t, mu, sig):
    u = torch.exp(
        -((t[None, None, :] - mu[:, None, None]) ** 2)
        / (2 * sig[None, :, None] ** 2)
    )
    v = u / torch.trapz(u, t, dim=-1).unsqueeze(-1)
    return v


def nonbatch_invert(*, domain, codomain):
    if len(domain.shape) != 1 or len(codomain.shape) != 1:
        raise ValueError(
            "domain and codomain must be 1D...use batch_invert instead"
        )
    if len(domain) != len(codomain):
        raise ValueError("domain and codomain must have the same length")
    return RGI(codomain, domain)


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = DotDict(cfg)
    for k in c.linspace.keys():
        c[k] = torch.linspace(*c[f'linspace.{k}'])
    return c


def process_cfg(c: DotDict) -> DotDict:
    c.gauss = gauss(t=c.t, mu=c.mu, sig=c.sig)
    c.quantile = torch.empty(*c.gauss.shape)
    for i in range(c.gauss.shape[0]):
        for j in range(c.gauss.shape[1]):
            input(c.t)
            input(c.gauss[i, j])
            input(c.p)
            c.quantile[i, j] = nonbatch_invert(
                domain=c.t, codomain=c.gauss[i, j]
            )(c.p)


def plotter(*, data, idx, fig, axes, lcls):
    plt.clf()
    plt.subplot(*lcls.shape, 1)
    plt.plot(data.t, data.gauss[idx])
    plt.title('Original')

    plt.subplot(*lcls.shape, 2)
    plt.plot(data.t, data.quantile[idx])
    plt.title('Quantile')
    plt.tight_layout()


def postprocess_cfg(c: DotDict) -> None:
    iter = bool_slice(*c.gauss.shape, none_dims=[-1])

    shape = (2, 1)
    kw_subplots = {'figsize': (12, 8)}
    fig, axes = plt.subplots(*shape, **kw_subplots)

    frames = get_frames_bool(
        data=c, iter=iter, fig=fig, axes=axes, plotter=plotter, lcls=locals()
    )
    save_frames(frames, path='out.gif')


@hydra.main(config_path='cfg', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    c = process_cfg(c)
    postprocess_cfg(c)


if __name__ == "__main__":
    start_time = time()
    main()
    total_time = time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
