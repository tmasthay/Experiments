import hydra
import torch
import deepwave
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from mh.core import DotDict as DD, DotDictImmutable as DDI


def get_velocity(model, shape, device):
    if isinstance(model, str):
        v = torch.load(model, map_location=device)
    elif isinstance(model, (int, float)):
        v = torch.full(shape, model, device=device, dtype=torch.float32)
    elif isinstance(model, list):
        v = torch.tensor(model, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported velocity model format: {type(model)}")

    v.requires_grad_(False)  # we are not optimizing velocity in this task
    return v


def cp(*args, device):
    grids = [torch.linspace(start, end, num) for start, end, num in args]
    return torch.cartesian_prod(*grids).to(device)


def rel_cp(*args, device):
    grids = [
        torch.linspace(start * dx, end * dx, num)
        for dx, start, end, num in args
    ]
    return torch.cartesian_prod(*grids).to(device)


def rel_cp_int(*args, device):
    u = rel_cp(*args, device=device)
    v = u.long()
    # remove duplicates
    v = torch.unique(v, dim=0)
    return v


def preprocess_cfg(cfg: DictConfig):
    c = DD(OmegaConf.to_container(cfg, resolve=True))
    c.source.peak_time = c._tmp_.peak_time_factor / c.simulation.pml_freq
    c.init_loc = [
        c._tmp_.init_loc[0] * c.grid.ny,
        c._tmp_.init_loc[1] * c.grid.nx,
    ]
    c.ref_loc = [c._tmp_.ref_loc[0] * c.grid.ny, c._tmp_.ref_loc[1] * c.grid.nx]
    c.grid.shape = [c.grid.ny, c.grid.nx]
    c.receivers.locations = rel_cp_int(*c.receivers.locations, device=c.device)[
        None, :, :
    ]
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


@hydra.main(config_path="all/gpt", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Preprocess configuration
    c, rt = preprocess_cfg(cfg)

    device = c.device

    # Load or initialize the wavespeed (velocity) model
    # Assuming cfg contains necessary fields or file paths for velocity
    v = get_velocity(c.velocity, c.grid.shape, device)

    # Simulation parameters from config
    nx, ny = c.grid.nx, c.grid.ny  # grid dimensions (50 x 50)
    dt = c.simulation.dt  # time step interval
    nt = c.simulation.nt  # number of time steps
    num_sources = c.num_sources  # e.g., 50
    # Gaussian initial parameters
    mu_x0 = c.source.mu_x
    mu_y0 = c.source.mu_y
    sigma_x0 = c.source.sigma_x
    sigma_y0 = c.source.sigma_y

    # Prepare receiver locations (assuming these are provided or configured)
    # If cfg contains receiver geometry (e.g., number and positions):
    assert c.receivers.locations.dim() == 3, (
        "Receiver locations should be 3D tensor (n_shots, n_receivers, 2), got"
        f" {c.receivers.locations.shape=}"
    )

    # Generate source wavelet (e.g., Ricker) to use for all sources
    freq = c.source.freq
    peak_time = c.source.peak_time
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)  # shape (nt,)
    wavelet = wavelet.to(device)  # move to device for simulation

    ref_src_loc = torch.tensor(c.ref_loc, dtype=torch.long, device=device)[
        None, None, :
    ]  # shape (1, 1, 2)
    ref_src_amp = wavelet.unsqueeze(0).unsqueeze(0)  # shape (1, 1, nt)
    observed_data = deepwave.scalar(
        v,
        c.grid.spacing,
        dt,
        source_amplitudes=ref_src_amp,
        source_locations=ref_src_loc,
        receiver_locations=c.receivers.locations,
        pml_freq=c.simulation.pml_freq,  # use PML frequency from config (if provided)
    )[-1]

    # Define the objective function that given (mu_x, mu_y, sigma_x, sigma_y) computes misfit
    def misfit(params: np.ndarray) -> float:
        mu_x, mu_y, sigma_x, sigma_y = params.astype(float)
        mu_x_t = torch.tensor(mu_x, dtype=torch.float32, device=device)
        mu_y_t = torch.tensor(mu_y, dtype=torch.float32, device=device)
        sigma_x_t = torch.tensor(sigma_x, dtype=torch.float32, device=device)
        sigma_y_t = torch.tensor(sigma_y, dtype=torch.float32, device=device)

        # Create coordinate grids for the 50x50 area
        x_coords = torch.arange(nx, device=device, dtype=torch.float32)
        y_coords = torch.arange(ny, device=device, dtype=torch.float32)
        Y_grid, X_grid = torch.meshgrid(
            y_coords, x_coords, indexing='ij'
        )  # shape (ny, nx)

        # Compute Gaussian weight at each grid point
        exp_arg = ((X_grid - mu_x_t) ** 2 / (2 * sigma_x_t**2)) + (
            (Y_grid - mu_y_t) ** 2 / (2 * sigma_y_t**2)
        )
        weight_grid = torch.exp(-exp_arg)
        # Apply hard cutoff outside 3-sigma ellipse
        mask = ((X_grid - mu_x_t) / sigma_x_t) ** 2 + (
            (Y_grid - mu_y_t) / sigma_y_t
        ) ** 2 <= 9.0  # 3-sigma support
        weight_grid = weight_grid * mask.float()

        # Flatten the weights and select top-K strongest points
        weight_flat = weight_grid.view(-1)  # length nx*ny
        # Get top `num_sources` values and their indices (in descending order of weight)
        topk_vals, topk_idx = torch.topk(
            weight_flat, k=num_sources, largest=True, sorted=True
        )
        # Convert flat indices to 2D grid indices
        topk_y = topk_idx // nx  # integer division to get row (y-index)
        topk_x = topk_idx % nx  # remainder to get col (x-index)
        # Stack into (x,y) coordinates for each source
        coords = torch.stack(
            (topk_x, topk_y), dim=1
        ).long()  # shape (num_sources, 2)
        coords = coords.unsqueeze(0).to(
            device
        )  # shape (1, num_sources, 2) for one shot

        # Construct source amplitude tensor for Deepwave: shape [1, num_sources, nt]
        # Scale the base wavelet for each source by that source's Gaussian weight
        # `topk_vals` are the weights for each selected source (length num_sources)
        # Expand and multiply to get a full time series per source:
        source_amplitudes = topk_vals.unsqueeze(1) * wavelet.unsqueeze(
            0
        )  # shape (num_sources, nt)
        source_amplitudes = source_amplitudes.unsqueeze(0).to(
            device
        )  # shape (1, num_sources, nt)

        # Run Deepwave forward modeling with these sources
        out = deepwave.scalar(
            v,
            c.grid.spacing,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=coords,
            receiver_locations=c.receivers.locations,
            pml_freq=c.simulation.pml_freq,  # use PML frequency from config (if provided)
        )
        # Deepwave returns a tuple; the last element is the receivers' recorded data&#8203;:contentReference[oaicite:6]{index=6}
        simulated_data = out[-1]  # shape: [n_shots, n_receivers, nt]

        # Compute mean squared error between simulated and observed data
        # (Assume observed_data has shape [n_shots, n_receivers, nt] matching simulated_data)
        mse_loss = torch.nn.functional.mse_loss(simulated_data, observed_data)
        return float(mse_loss.item())  # return as Python float for SciPy

    # Initial parameter vector for Nelder-Mead
    x0 = np.array([mu_x0, mu_y0, sigma_x0, sigma_y0], dtype=float)
    # Run Nelder-Mead optimization to minimize the misfit

    def printing_callback(xk):
        printing_callback.iteration += 1
        current_misfit = misfit(xk)
        true_dist = np.linalg.norm(
            np.array([c.ref_loc[0] - xk[0], c.ref_loc[1] - xk[1]])
        )
        print(
            f"Iteration {printing_callback.iteration}: parameters = {xk}"
            f", misfit = {current_misfit}"
            f", ground_truth = {c.ref_loc}"
            f", true_distance = {true_dist:.2e}"
        )

    printing_callback.iteration = 0
    result = minimize(
        misfit,
        x0,
        method='Nelder-Mead',
        options={'maxiter': c.optim.maxiter, 'disp': True},
        callback=printing_callback,
    )

    # Output the optimization results
    optimized_params = (
        result.x
    )  # [mu_x, mu_y, sigma_x, sigma_y] that minimize the misfit
    print(
        f"Optimized Gaussian parameters: mu_x={optimized_params[0]:.3f},"
        f" mu_y={optimized_params[1]:.3f}, sigma_x={optimized_params[2]:.3f},"
        f" sigma_y={optimized_params[3]:.3f}"
    )
    print(f"Final misfit value: {result.fun:.6f}")


if __name__ == "__main__":
    main()
