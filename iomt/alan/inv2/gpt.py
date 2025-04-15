import functools
import hydra
import torch
import deepwave
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from mh.core import DotDict as DD, DotDictImmutable as DDI
from helpers import l2_loss, eff_quasi_w1_loss

import sys

class Tee:
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self.file.close()
        
def tee_output(filename, mode="w"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Tee(filename, mode):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def hydra_tee(func):
    @functools.wraps(func)
    def wrapper(cfg, *args, **kwargs):
        dupe_filename = getattr(cfg, "dupe", None)
        if dupe_filename:
            with Tee(dupe_filename):
                return func(cfg, *args, **kwargs)
        else:
            return func(cfg, *args, **kwargs)
    return wrapper

# Example usage:
# with Tee("output.log"):
#     print("This message will go to both stdout and the file.")

def cp(*args, device):
    grids = [torch.linspace(start, end, num) for start, end, num in args]
    return torch.cartesian_prod(*grids).to(device)


def rel_cp(*args, device):
    grids = [torch.linspace(start * dx, end * dx, num) for dx, start, end, num in args]
    return torch.cartesian_prod(*grids).to(device)


def rel_cp_int(*args, device):
    pts = rel_cp(*args, device=device)
    return torch.unique(pts.long(), dim=0)


def preprocess_cfg(cfg: DictConfig):
    c = DD(OmegaConf.to_container(cfg, resolve=True))
    c.source.peak_time = c._tmp_.peak_time_factor / c.simulation.pml_freq
    c.init_loc = [c._tmp_.init_loc[0] * c.grid.ny, c._tmp_.init_loc[1] * c.grid.nx]
    c.ref_loc = [c._tmp_.ref_loc[0] * c.grid.ny, c._tmp_.ref_loc[1] * c.grid.nx]
    c.grid.shape = [c.grid.ny, c.grid.nx]
    c.receivers.locations = rel_cp_int(*c.receivers.locations, device=c.device)[None, :, :]
    if c.device.startswith('cuda') and torch.cuda.is_available():
        c.device = torch.device(c.device)
    else:
        c.device = torch.device('cpu')
    del c._tmp_

    if c.num_sources == 'all':
        c.num_sources = c.grid.nx * c.grid.ny - 4
    c = DDI(c)
    rt = DD({})
    return c, rt


def get_velocity(model, shape, device):
    if isinstance(model, str):
        v = torch.load(model, map_location=device)
    elif isinstance(model, (int, float)):
        v = torch.full(shape, model, device=device, dtype=torch.float32)
    elif isinstance(model, list):
        v = torch.tensor(model, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported velocity model format: {type(model)}")
    v.requires_grad_(False)
    return v


@hydra.main(config_path="all/gpt", config_name="default", version_base=None)
@hydra_tee
def main(cfg: DictConfig):
    # Preprocess configuration (keep as-is)
    c, rt = preprocess_cfg(cfg)
    device = c.device
    v = get_velocity(c.velocity, c.grid.shape, device)

    nx, ny = c.grid.nx, c.grid.ny
    dt, nt = c.simulation.dt, c.simulation.nt
    num_sources = c.num_sources

    # Generate source wavelet and send to device.
    wavelet = deepwave.wavelets.ricker(c.source.freq, nt, dt, c.source.peak_time).to(device)
    receivers = c.receivers.locations

    # Create observed data using a reference source.
    ref_src_loc = torch.tensor(c.ref_loc, dtype=torch.long, device=device)[None, None, :]
    ref_src_amp = wavelet.unsqueeze(0).unsqueeze(0)
    observed_data = deepwave.scalar(
        v,
        c.grid.spacing,
        dt,
        source_amplitudes=ref_src_amp,
        source_locations=ref_src_loc,
        receiver_locations=receivers,
        pml_freq=c.simulation.pml_freq,
    )[-1]

    # Optimize parameters: [mu_x, mu_y, log(sigma_x), log(sigma_y)]
    x0 = np.array(
        [c.source.mu_x, c.source.mu_y, np.log(c.source.sigma_x), np.log(c.source.sigma_y)],
        dtype=float,
    )

    loss = eff_quasi_w1_loss(observed_data, torch.nn.functional.softplus)
    
    def misfit(params: np.ndarray) -> float:
        mu_x, mu_y, log_sigma_x, log_sigma_y = params.astype(float)
        # Constrain sigma to be positive via exp transform.
        sigma_x = torch.exp(torch.tensor(log_sigma_x, device=device, dtype=torch.float32))
        sigma_y = torch.exp(torch.tensor(log_sigma_y, device=device, dtype=torch.float32))
        mu_x_t = torch.tensor(mu_x, device=device, dtype=torch.float32)
        mu_y_t = torch.tensor(mu_y, device=device, dtype=torch.float32)

        # Construct 2D grid.
        x_coords = torch.arange(nx, device=device, dtype=torch.float32)
        y_coords = torch.arange(ny, device=device, dtype=torch.float32)
        Y_grid, X_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Gaussian weight with a 3-sigma cutoff.
        exp_arg = ((X_grid - mu_x_t) ** 2 / (2 * sigma_x ** 2)) + ((Y_grid - mu_y_t) ** 2 / (2 * sigma_y ** 2))
        weight_grid = torch.exp(-exp_arg)
        
        do_mask = False
        if do_mask: 
            mask = (((X_grid - mu_x_t) / sigma_x) ** 2 + ((Y_grid - mu_y_t) / sigma_y) ** 2 <= 9.0)
            weight_grid *= mask.float()

            weight_flat = weight_grid.view(-1)
            topk_vals, topk_idx = torch.topk(weight_flat, k=num_sources, largest=True, sorted=True)
            topk_y = topk_idx // nx
            topk_x = topk_idx % nx
            coords = torch.stack((topk_x, topk_y), dim=1).long().unsqueeze(0).to(device)
            src_amp = (topk_vals.unsqueeze(1) * wavelet.unsqueeze(0)).unsqueeze(0).to(device)
        else:
            src_amp = weight_grid.view(-1).unsqueeze(0) * wavelet.unsqueeze(0)
            
            # coords are just the original source locations since no filtering
            coords = cp(
                (0, nx - 1, nx), (0, ny - 1, ny), device=device
            ).long().unsqueeze(0)

        sim_data = deepwave.scalar(
            v,
            c.grid.spacing,
            dt,
            source_amplitudes=src_amp,
            source_locations=coords,
            receiver_locations=receivers,
            pml_freq=c.simulation.pml_freq,
        )[-1]
        # return float(l2_loss(sim_data, observed_data))
        return float(loss(sim_data))

    def callback_wrapper():
        num_calls = 0
        def callback(xk):
            nonlocal num_calls
            num_calls += 1
            print(f"Iteration: {num_calls}, Params: [{xk[0]},{xk[1]},{np.exp(xk[2])},{np.exp(xk[3])}], Misfit: {misfit(xk)}")
        return callback

    result = minimize(
        misfit,
        x0,
        method="Nelder-Mead",
        callback=callback_wrapper(),
        options={"maxiter": c.optim.maxiter, "disp": True},
    )

    opt = result.x
    print(
        f"Optimized: mu_x={opt[0]}, mu_y={opt[1]}, sigma_x={np.exp(opt[2])}, sigma_y={np.exp(opt[3])}"
    )
    print(f"Final misfit: {result.fun}")


if __name__ == "__main__":
    main()