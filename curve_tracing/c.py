from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
import deepwave as dw
from misfit_toys.data.dataset import towed_src, fixed_rec
from mh.typlotlib import save_frames, get_frames_bool
from misfit_toys.utils import bool_slice, clean_idx
import hydra
from dotmap import DotMap


def get_grid(a, b, c, d, M, N):
    x = torch.linspace(a, b, M)
    y = torch.linspace(c, d, N)
    return torch.stack(torch.meshgrid(x, y, indexing='ij'), 2).view(-1, 2)


def hyperbola(
    *, t0, t1, N, a, b, center, box_constraint=None, flip=False, keep_dims=True
):
    t = torch.linspace(t0, t1, N)
    if flip:
        v = torch.stack(
            [center[0] + b * torch.sinh(t), center[1] + a * torch.cosh(t)], 1
        )
    else:
        v = torch.stack(
            [center[0] + a * torch.cosh(t), center[1] + b * torch.sinh(t)], 1
        )
    if box_constraint is not None:
        indices = (
            (v[:, 0] >= box_constraint[0])
            & (v[:, 0] <= box_constraint[1])
            & (v[:, 1] >= box_constraint[2])
            & (v[:, 1] <= box_constraint[3])
        )
        v = v[indices]
        if keep_dims:
            new_len = v.shape[0]
            # pad with same value as v[-1] at the end
            v = torch.cat([v, v[-1].unsqueeze(0).expand(N - new_len, 2)], 0)
    return v


def triangle_line(p0, p1, p2, N):
    t = torch.linspace(0, 1, N)
    v = p0 + t[:, None] * (p1 - p0) + t[:, None] * (p2 - p0)
    return v


def proj_indices(X, Y, K):
    neigh = NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
    neigh.fit(X)
    distances, indices = neigh.kneighbors(Y)
    return distances, indices


def proj_function(
    function_vals: torch.Tensor,
    domain: torch.Tensor,
    embedding: torch.Tensor,
    K: int,
):
    distances, indices = proj_indices(domain, embedding, K)

    weights = torch.tensor(1.0 / (1e-6 + distances), dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    indices = torch.tensor(indices, dtype=torch.int64)
    assert (
        len(function_vals.shape) == 1
    ), "function_vals must be 1D...flatten it"
    F = [[function_vals[ii] for ii in i] for i in indices]
    F = torch.tensor(F, dtype=torch.float32)
    weighted_vals = F * weights
    return weighted_vals.sum(dim=1), weights, indices


def main1():
    N = 25
    K = 3
    grid = get_grid(-2, 2, -2, 2, 10, 10)
    # Y = hyperbola(t0=-1, t1=1, N=N, a=-1, b=1, flip=True)
    Y = grid[:N]
    _, indices = proj_indices(grid, Y, K)
    function_vals = torch.rand(N)
    res = proj_function(function_vals, grid, Y, K)

    plt.scatter(grid[:, 0], grid[:, 1], c='blue', label='X')
    plt.scatter(Y[:, 0], Y[:, 1], c='red', label='Y')
    for i in range(N):
        for j in range(K):
            plt.plot(
                [Y[i, 0], grid[indices[i, j], 0]],
                [Y[i, 1], grid[indices[i, j], 1]],
                'k-',
            )
    plt.savefig('scatter.png')

    plt.clf()
    plt.plot(range(res.shape[0]), res)
    plt.savefig('function.png')


def main2():
    domain = get_grid(-2, 2, -2, 2, 100, 100)
    embedding = hyperbola(t0=-1, t1=1, N=25, a=-1, b=1, flip=True)
    # embedding = domain[:25]
    function_vals = torch.rand(domain.shape[0])
    K = 10
    Z, _, indices = proj_function(function_vals, domain, embedding, K)
    print(Z.shape)
    plt.plot(range(Z.shape[0]), Z, 'b-', label='projected')
    plt.plot(range(Z.shape[0]), torch.rand(Z.shape[0]), 'r--', label='random')
    plt.savefig('function.png')

    plt.clf()
    plt.scatter(domain[:, 0], domain[:, 1], c='blue', label='X')
    plt.scatter(embedding[:, 0], embedding[:, 1], c='red', label='Y')
    for i in range(embedding.shape[0]):
        for j in range(K):
            plt.plot(
                [embedding[i, 0], domain[indices[i, j], 0]],
                [embedding[i, 1], domain[indices[i, j], 1]],
                'k-',
            )
    plt.savefig('scatter.png')


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(c: DictConfig):
    T = c.nt * c.dt

    vel = c.velocity * torch.ones(c.ny, c.nx)

    src_loc_y = towed_src(**c.src)
    rec_loc_y = fixed_rec(**c.rec)
    src_amp_y = dw.wavelets.ricker(
        freq=c.wavelet.freq,
        length=c.nt,
        dt=c.dt,
        peak_time=c.wavelet.peak_factor / c.wavelet.freq,
    ).expand(c.n_shots, c.src_per_shot, c.nt)

    u = dw.scalar(
        vel,
        grid_spacing=[c.dy, c.dx],
        dt=c.dt,
        source_amplitudes=src_amp_y,
        source_locations=src_loc_y,
        receiver_locations=rec_loc_y,
    )

    obs_data = u[-1]

    b_vals = torch.linspace(10, 800, 10)
    v = []
    for b in b_vals:
        v.append(
            hyperbola(
                t0=-1,
                t1=1,
                N=c.N,
                a=1.0,
                b=b,
                center=[200, -0.95],
                flip=True,
                box_constraint=[0.0, c.rec_per_shot * c.dy, 0.0, T],
            )
        )
    v = torch.stack(v, dim=0)

    w = []
    domain = get_grid(0, c.rec_per_shot * c.dy, 0, T, c.rec_per_shot, c.nt)
    for vv in v:
        res, _, _ = proj_function(
            function_vals=obs_data[obs_data.shape[0] // 2].reshape(-1),
            domain=domain,
            embedding=vv,
            K=c.K,
        )
        w.append(res)

    w = torch.stack(w, dim=0)

    def plotter(*, data, idx, fig, axes):
        plt.clf()

        plt.subplot(*c.plt.sub.shape, 1)
        plt.imshow(
            data.u.T,
            extent=[0, c.rec_per_shot * c.dy, T, 0],
            **c.plt.get('imshow', {}),
        )
        plt.colorbar()
        plt.title(f'{c.K=} idx={clean_idx(idx)}')
        plt.xlabel('Receiver Location (m)')
        plt.ylabel('Time (s)')
        plt.scatter(
            data.v[idx[0], :, 0],
            v[idx[0], :, 1],
            c='blue',
            label='hyperbola',
            s=4,
        )

        plt.subplot(*c.plt.sub.shape, 2)
        plt.plot(w[idx[0]])
        plt.ylim([obs_data.min(), obs_data.max()])
        plt.title(f'Trace along curve at idx={clean_idx(idx)}')

    if c.plt.get('go', True):
        # u = obs_data[obs_data.shape[0] // 2].unsqueeze(0)
        u = obs_data[obs_data.shape[0] // 2]
        data = DotMap({'u': u, 'v': v})
        iter = bool_slice(v.shape[0], **c.plt.iter)
        fig, axes = plt.subplots(*c.plt.sub.shape, **c.plt.sub.kw)
        frames = get_frames_bool(
            data=data,
            iter=iter,
            plotter=plotter,
            fig=fig,
            axes=axes,
            framer=None,
        )
        save_frames(frames, path=f'res{c.K}')


if __name__ == '__main__':
    main()
