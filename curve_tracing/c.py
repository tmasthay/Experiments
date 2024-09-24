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


def proj_indices_legacy(X, Y, K):
    neigh = NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
    neigh.fit(X)
    distances, indices = neigh.kneighbors(Y)
    return distances, indices


def proj_indices(
    X: torch.Tensor,
    Y: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-6,
    alpha: float = 1.0,
):
    neigh = NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
    neigh.fit(X)
    distances, indices = neigh.kneighbors(Y)
    weights = torch.tensor(
        1.0 / (eps + distances) ** alpha, dtype=torch.float32
    )
    weights = weights / weights.sum(dim=1, keepdim=True)
    indices = torch.tensor(indices, dtype=torch.int64)
    return weights, indices


def weighted_average(
    *, values: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
):
    # refactor if performance is an issue
    # list comprehension is slow compared to torch operations

    assert len(values.shape) == 1, "values must be 1D...flatten it"
    assert values.dtype in [
        torch.float32,
        torch.float64,
    ], "values must be float"
    assert values.shape[0] == indices.shape[0]
    assert indices.dtype in [torch.int32, torch.int64]
    assert indices.min() >= 0 & indices.max() < values.shape[0]
    assert weights.shape == indices.shape

    F = torch.gather(
        values.unsqueeze(1).expand(-1, indices.size(1)), 1, indices
    )
    weighted_vals = F * weights
    return weighted_vals.sum(dim=1)


def proj_function(
    *,
    function_vals: torch.Tensor,
    domain: torch.Tensor,
    embedding: torch.Tensor,
    num_neighbors: int,
    eps: float = 1e-6,
    alpha: float = 1.0,
):
    assert (
        len(function_vals.shape) == 1
    ), "function_vals must be 1D...flatten it"

    weights, indices = proj_indices(
        domain, embedding, num_neighbors, eps=eps, alpha=alpha
    )

    weighted_vals = weighted_average(
        values=function_vals, weights=weights, indices=indices
    )
    return weighted_vals.sum(dim=1), weights, indices


def batch_proj_function(
    *,
    function_vals: torch.Tensor,
    domain: torch.Tensor,
    embeddings: torch.Tensor,
    num_neighbors: int,
    eps: float = 1e-6,
    alpha: float = 1.0,
    device: str = "cpu",
):
    res = [None for _ in range(embeddings.shape[0])]
    weights = [None for _ in range(embeddings.shape[0])]
    indices = [None for _ in range(embeddings.shape[0])]
    for i, embedding in enumerate(embeddings):
        res[i], weights[i], indices[i] = proj_function(
            function_vals=function_vals,
            domain=domain,
            embedding=embedding,
            num_neighbors=num_neighbors,
            eps=eps,
            alpha=alpha,
        )
    res = torch.stack(res, 0).to(device)
    weights = torch.stack(weights, 0).to(device)
    indices = torch.stack(indices, 0).to(device)
    return res, weights, indices


class SlicedWasserstein(torch.nn.Module):
    def __init__(
        self,
        *,
        data,
        domain,
        embeddings,
        num_neighbors=1,
        alpha=1.0,
        eps=1e-6,
        device,
    ):
        super().__init__()

        self.data, self.weights, self.indices = batch_proj_function(
            function_vals=data,
            domain=domain,
            embeddings=embeddings,
            num_neighbors=num_neighbors,
            eps=eps,
            alpha=alpha,
            device=device,
        )

    def forward(self, x):
        return x
