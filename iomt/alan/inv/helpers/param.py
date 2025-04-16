import os
import torch
import deepwave as dw
import matplotlib.pyplot as plt
from mh.typlotlib import save_frames, get_frames_bool, bool_slice
from torch.nn import functional as F
from mh.core import DotDictImmutable as DDI
from rl_batch import BatchedRiemannLiouvilleFractionalIntegral as RLInt


class PiecewiseAlphaScheduler:
    def __init__(self, num_alphas, step_size, device):
        self.num_alphas = num_alphas
        self.step_size = step_size
        self.call_no = 0
        self.device = device

    def __call__(self):
        self.call_no += 1
        index = min(self.call_no // self.step_size, self.num_alphas - 1)
        weights = torch.zeros(self.num_alphas)
        weights[index] = 1.0
        return weights.to(self.device)


def rl_loss(observed_data, alphas, alpha_weights, dt, max_length, gamma=0.1):
    """
    Compute the Riemann-Liouville fractional integral loss for a batch of data.

    Parameters:
        alphas: 1D list or torch.Tensor of scalars (shape [num_alphas]).
        dt: time sampling interval.
        max_length: maximum length of the input signal for precomputing the kernels.
        gamma: a small constant to avoid singularity at 0.

    Returns:
        A function that computes the loss given simulated and observed data.
    """
    preprocess = torch.nn.Softplus(beta=1.0, threshold=20.0)
    rl_int = RLInt(alphas=alphas, dt=dt, max_length=max_length, gamma=gamma)
    filtered_obs = preprocess(observed_data)

    def helper(sim_data):
        # Compute the Riemann-Liouville fractional integral for both simulated and observed data.
        filtered_sim_data = preprocess(sim_data)
        sim_data_rl = rl_int(filtered_sim_data - filtered_obs)
        res = alpha_weights()[:, None] * sim_data_rl
        return (sim_data_rl**2).mean()

    return helper


class IdentityMomentSource(torch.nn.Module):
    def __init__(
        self,
        *,
        src_loc,
        mu: torch.Tensor,
        sig: torch.Tensor,
        peak_time: torch.Tensor,
        freq: torch.Tensor,
        scale: torch.Tensor,
        nt: int,
        dt: float,
        device: str = "cpu",
    ):
        super().__init__()

        self.src_loc = src_loc
        self.mu = torch.nn.Parameter(mu, requires_grad=mu.requires_grad).to(
            device
        )
        self.sig = torch.nn.Parameter(sig, requires_grad=sig.requires_grad).to(
            device
        )
        self.peak_time = torch.nn.Parameter(
            peak_time, requires_grad=peak_time.requires_grad
        ).to(device)
        self.freq = torch.nn.Parameter(
            freq, requires_grad=freq.requires_grad
        ).to(device)
        self.scale = torch.nn.Parameter(
            scale, requires_grad=scale.requires_grad
        ).to(device)
        self.device = device
        self.num_shots = self.src_loc.shape[0]
        self.num_sources = self.src_loc.shape[1]
        self.nt = nt
        self.dt = dt
        self.target_shape = torch.Size(
            [self.num_shots, self.num_sources, self.nt]
        )

        self.__validate_parameters()

    def __validate_parameters(self):
        def check_elem_dim(t, dims, elems, name):
            assert t.ndim == dims, f"{name} must be {dims}D, got {t.ndim}D"
            assert (
                t.nelement() == elems
            ), f"{name} must have {elems} elements, got {t.nelement()}"

        assert (
            self.src_loc.ndim == 3
        ), f"src_loc must be 3D, got {self.src_loc.ndim}D"
        assert self.src_loc.shape[2] == 2, f"{self.src_loc.shape[1]=} != 2"
        assert (
            self.src_loc.dtype == torch.int
        ), f"src_loc must be int, got {self.src_loc.dtype}"
        check_elem_dim(self.mu, 1, 2, "mu")
        check_elem_dim(self.sig, 1, 2, "sig")
        check_elem_dim(self.peak_time, 0, 1, "peak_time")
        check_elem_dim(self.freq, 0, 1, "freq")
        check_elem_dim(self.scale, 0, 1, "scale")

        assert self.nt > 0, f"nt must be positive, got {self.nt=}"
        assert self.dt > 0, f"dt must be positive, got {self.dt=}"

        # check they are all on the right device
        def check_device(t, name):
            assert torch.device(t.device) == torch.device(
                self.device
            ), f"{name} must be on {self.device}, got {t.device}"

        check_device(self.mu, "mu")
        check_device(self.sig, "sig")
        check_device(self.peak_time, "peak_time")
        check_device(self.freq, "freq")
        check_device(self.scale, "scale")
        check_device(self.src_loc, "src_loc")

    def forward(self):
        wavelet = dw.wavelets.ricker(
            freq=self.freq.cpu(),
            length=self.nt,
            dt=self.dt,
            peak_time=self.peak_time.cpu(),
            dtype=torch.float32,
        ).to(self.device)
        arg = -torch.sum(
            (self.src_loc.float() - self.mu[None, None, :])
            / self.sig[None, None, :],
            dim=-1,
        )
        exp_term = torch.exp(arg)

        res = wavelet[None, None, :] * exp_term[:, :, None] * self.scale

        assert (
            res.shape == self.target_shape
        ), f"Expected shape {self.target_shape}, got {res.shape}"
        return res.to(self.device)


def l2_loss(sim_data, observed_data):
    return torch.nn.functional.mse_loss(sim_data, observed_data)


def captured_l2_loss(observed_data):
    def helper(sim_data):
        return F.mse_loss(sim_data, observed_data)

    return helper


def quasi_cdf(data, positive_routine):
    u1 = data.reshape(-1, data.shape[-1])  # always along last dim

    # renorm the data and then take cumsum along the last dim
    u2 = positive_routine(u1)
    u3 = torch.cumulative_trapezoid(u2, dim=-1)

    # renormalize based on final sum
    u4 = u3 / u3[..., -1].unsqueeze(-1)
    return u4


def quasi_w1_loss(sim_data, observed_data, positive_routine):
    a = quasi_cdf(sim_data, positive_routine)
    b = quasi_cdf(observed_data, positive_routine)
    return torch.nn.functional.mse_loss(a, b)


def eff_quasi_w1_loss(observed_data, positive_routine):
    norm_data = quasi_cdf(observed_data, positive_routine)

    def helper(sim_data):
        a = quasi_cdf(sim_data, positive_routine)
        return torch.nn.functional.mse_loss(a, norm_data)

    return helper


def get_grad_clone(t):
    return None if t.grad is None else t.grad.clone().cpu().numpy()


def get_hist(t):
    data_clone = t.detach().clone().cpu().numpy()
    grad_clone = get_grad_clone(t)
    return {"data": data_clone, "grad": grad_clone}


def main_big():
    device = "cuda:0"
    nt = 1000  # Number of time samples
    dt = 0.004  # Time sampling interval
    grid_spacing = [10.0, 10.0]
    # Define a simple velocity model
    nx, ny = 50, 50
    vp = torch.ones((ny, nx), device=device) * 1500.0  # Constant velocity model

    # Define source and receiver locations
    # For this example, we use one shot with one source at a fixed grid location.
    num_shots = 1
    num_sources = 1
    # Source location: shape [num_shots, num_sources, 2] (grid indices)
    src_loc = torch.tensor([[[25, 25]]]).int().to(device)
    # Define receiver locations: for example, receivers along the bottom row of the grid.
    # rec_locs = torch.stack(
    #     [torch.arange(0, nx).int(), torch.full((nx,), ny - 1, dtype=torch.int)],
    #     dim=-1
    # ).unsqueeze(0).to(device) # shape: [1, nx, 2]
    rec_locs = (
        torch.cartesian_prod(
            torch.tensor([2]).int(), torch.arange(0, nx, 5).int()
        )
        .unsqueeze(0)
        .to(device)
    )  # shape: [1, nx, 2]
    # Ground-truth source parameters
    true_mu = torch.tensor([25.0, 25.0]).to(device)
    true_sig = torch.tensor([0.1, 0.1]).to(device)
    true_peak_time = torch.tensor(0.2).to(device)
    true_freq = torch.tensor(25.0).to(device)
    true_scale = torch.tensor(1.0).to(device)

    # Create a ground-truth IdentityMomentSource module
    source_true = IdentityMomentSource(
        src_loc=src_loc,
        mu=true_mu,
        sig=true_sig,
        peak_time=true_peak_time,
        freq=true_freq,
        scale=true_scale,
        nt=nt,
        dt=dt,
        device=device,
    )

    # Generate synthetic source amplitudes using ground truth parameters.
    true_src_amp = source_true.forward()
    # Generate synthetic data via deepwave.scalar.
    # Here, we assume dw.scalar returns a tuple with the last element as the receiver data.
    syn_data = dw.scalar(
        vp,
        grid_spacing,
        dt,
        source_locations=src_loc,  # Should be provided in the expected format by dw.scalar.
        receiver_locations=rec_locs,
        source_amplitudes=true_src_amp,
        pml_width=10,
    )[-1]
    root_mean_square_syn = torch.sqrt(
        torch.mean(syn_data**2, dim=-1, keepdim=True)
    )  # shape: [num_shots, num_receivers, 1]
    noise_model = torch.randn_like(syn_data) * 0.1 * root_mean_square_syn
    syn_data = syn_data.detach() + noise_model

    # _loss = captured_l2_loss(syn_data)
    # _loss = eff_quasi_w1_loss(
    #     syn_data, torch.nn.Softplus(beta=1.0, threshold=20.0)
    # )
    alphas = [1.0]
    num_epochs = 10000
    step_size = num_epochs // len(alphas)
    alpha_weights = PiecewiseAlphaScheduler(
        num_alphas=len(alphas), step_size=step_size, device=device
    )
    _loss = rl_loss(
        alphas=[1.0],
        dt=dt,
        max_length=nt,
        gamma=0.1,
        observed_data=syn_data,
        alpha_weights=alpha_weights,
    )

    def get_msg(mu, sig, peak_time, freq, scale, version):
        arr = ['\n\n']
        arr.append(version)
        arr.append(f"mu: {mu.detach().cpu().numpy()}")
        arr.append(f"sig: {sig.detach().cpu().numpy()}")
        arr.append(f"peak_time: {peak_time.detach().cpu().item()}")
        arr.append(f"freq: {freq.detach().cpu().item()}")
        arr.append(f"scale: {scale.detach().cpu().item()}")
        arr.append('\n\n')
        return '\n'.join(arr)

    # Now, create a model with an initial guess for the parameters.
    init_mu = torch.tensor([20.0, 20.0]).to(device).requires_grad_(True)
    init_sig = torch.tensor([5.0, 5.0]).to(device).requires_grad_(True)
    init_peak_time = torch.tensor(0.23).to(device).requires_grad_(True)
    init_freq = torch.tensor(23.0).to(device).requires_grad_(True)
    init_scale = torch.tensor(2.0).to(device).requires_grad_(True)

    init_msg = get_msg(
        init_mu,
        init_sig,
        init_peak_time,
        init_freq,
        init_scale,
        'Initial guess:',
    )
    truth_msg = get_msg(
        true_mu,
        true_sig,
        true_peak_time,
        true_freq,
        true_scale,
        'Ground truth:',
    )

    source_model = IdentityMomentSource(
        src_loc=src_loc,
        mu=init_mu,
        sig=init_sig,
        peak_time=init_peak_time,
        freq=init_freq,
        scale=init_scale,
        nt=nt,
        dt=dt,
        device=device,
    )
    # input(list(source_model.parameters()))
    # Use Adam optimizer on the source_model parameters.
    optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-2)
    # optimizer = torch.optim.LBFGS(source_model.parameters())
    abs_loss_tol = 1e-8

    history = []
    history_freq = 10
    for epoch in range(num_epochs):
        def closure():
            optimizer.zero_grad()
            pred_src_amp = source_model.forward()
            pred_data = dw.scalar(
                vp,
                grid_spacing,
                dt,
                source_locations=src_loc,
                receiver_locations=rec_locs,
                source_amplitudes=pred_src_amp,
                pml_width=10,
            )[-1]
            loss = _loss(pred_data)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        curr_loss = loss.item()

        if curr_loss < abs_loss_tol:
            print(f"Converged at epoch {epoch} with loss {curr_loss:.6f}")
            break

        if epoch % history_freq == 0 or epoch == num_epochs - 1:
            history.append(
                DDI(
                    {
                        "epoch": epoch,
                        "mu": get_hist(source_model.mu),
                        "sig": get_hist(source_model.sig),
                        "peak_time": get_hist(source_model.peak_time),
                        "freq": get_hist(source_model.freq),
                        "scale": get_hist(source_model.scale),
                        "loss": curr_loss,
                    }
                )
            )

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch} | Loss: {curr_loss:.6e}")
            print(
                f"mu: {source_model.mu.data.cpu().numpy()}, "
                f"sig: {source_model.sig.data.cpu().numpy()}, "
                f"peak_time: {source_model.peak_time.item()}, "
                f"freq: {source_model.freq.item()}, "
                f"scale: {source_model.scale.item()}"
            )

    final_res_msg = get_msg(
        source_model.mu,
        source_model.sig,
        source_model.peak_time,
        source_model.freq,
        source_model.scale,
        'Final result:',
    )

    full_msg = f"{init_msg}\n{truth_msg}\n{final_res_msg}"
    print(full_msg)


# Minimal working example
def main():
    device = "cpu"
    num_shots = 1
    num_sources = 3
    nt = 100
    dt = 0.004

    # src_loc: Each source has a coordinate in 2D. Shape: [num_shots, num_sources, 2]
    src_loc = torch.tensor([[[10.0, 20.0], [15.0, 20.5], [11.0, 25.0]]]).int()
    # Set mu and sig as 1D tensors of length 2 (for x and y) shifts/scaling.
    mu = torch.tensor([10.0, 20.0])
    sig = torch.tensor([5.0, 1.0])
    # peak_time, freq, and scale are scalars (set as 0D tensors for parameters)
    peak_time = torch.tensor(0.2)
    freq = torch.tensor(25.0)
    scale = torch.tensor(10.0)

    # Instantiate the module.
    source_module = IdentityMomentSource(
        src_loc=src_loc,
        mu=mu,
        sig=sig,
        peak_time=peak_time,
        freq=freq,
        scale=scale,
        nt=nt,
        dt=dt,
        device=device,
    )

    # Run a forward pass.
    output = source_module.forward()
    print("Output shape:", output.shape)  # Expect [1, 3, 100]

    # Plot the output waveforms for each source in the single shot.
    style = ['--', '-', ':']
    for j in range(output.shape[1]):
        plt.plot(
            output[0, j, :].detach().cpu().numpy(),
            label=f"Source {j}",
            linestyle=style[j],
        )
    plt.xlabel("Time sample")
    plt.ylabel("Amplitude")
    plt.title("IdentityMomentSource Output Waveforms (Shot 0)")
    plt.legend()
    plt.savefig('out.jpg')

    print('\n\nout.jpg\n\n')


def main2d():
    device = "cpu"
    nt = 100  # Number of time samples in the wavelet
    dt = 0.1  # Time sampling interval
    num_shots = 1  # Single shot for visualization

    # Define a grid in the normalized [0, 1] range for both coordinates.
    grid_size = 25
    xs = torch.linspace(0, 1, grid_size)
    ys = torch.linspace(0, 1, grid_size)
    # Create a grid of points (each point is 2D) using cartesian product.
    grid_points = torch.cartesian_prod(xs, ys)  # shape: [num_points, 2]
    # For compatibility, reshape to have a batch dimension (num_shots x num_sources x 2).
    src_loc = grid_points.unsqueeze(0)  # shape: [1, grid_size^2, 2]

    # Set parameters:
    # Let mu be somewhere near the center of the [0,1] interval in both directions.
    mu = torch.tensor([0.5, 0.5])
    # sig controls the spread. Pick a moderate value.
    sig = torch.tensor([100.0, 100.0])
    # Choose peak_time, freq and scale for the Ricker wavelet.
    peak_time = torch.tensor(dt * nt / 4)  # Midpoint in time
    freq = torch.tensor(2.0)
    scale = torch.tensor(1.0)

    # Instantiate the module.
    source_module = IdentityMomentSource(
        src_loc=src_loc,
        mu=mu,
        sig=sig,
        peak_time=peak_time,
        freq=freq,
        scale=scale,
        nt=nt,
        dt=dt,
        device=device,
    )

    # Run a forward pass.
    output = (
        source_module.forward().detach().cpu().numpy()
    )  # shape: [1, num_sources, nt]
    print("Output shape:", output.shape)

    output = output.squeeze().reshape(
        grid_size, grid_size, nt
    )  # shape: [grid_size, grid_size, nt]
    # def plotter(*, data, idx, fig, axes):
    #     plt.clf()
    #     plt.imshow(output[:, :, idx[0]].squeeze().reshape(grid_size, grid_size), cmap='viridis', origin='lower',
    #             extent=(0, 1, 0, 1), vmin=output.min(), vmax=output.max(), aspect='auto')
    #     plt.colorbar(label="Amplitude")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title(f"t={idx[0]*dt:.3f}s")

    # fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    # iter = bool_slice(nt, strides=[nt // 20])
    # frames = get_frames_bool(data=None, iter=iter, plotter=plotter)
    # save_frames(frames, path='movie', duration=1000)

    # def plotter(*, data, idx, fig, axes):
    #     plt.clf()
    #     plt.imshow(output[idx].squeeze().reshape(grid_size, nt), cmap='viridis', origin='lower',
    #             extent=(0, nt*dt, 0, grid_size*dt), vmin=output.min(), vmax=output.max(), aspect='auto')
    #     plt.colorbar(label="Amplitude")
    #     plt.xlabel("t")
    #     plt.ylabel("y")
    #     plt.title(f"x={idx[1]*dt:.3f}s")

    # iter = bool_slice(*output.shape, strides=[1, grid_size // 20, 1], none_dims=[1])
    # frames = get_frames_bool(data=None, iter=iter, plotter=plotter, fig=fig, axes=axes)
    # save_frames(frames, path='movie_yt', duration=1000)

    def plotter(*, data, idx, fig, axes):
        plt.clf()
        plt.plot(data[idx[0], idx[1], :])
        plt.title(f"({idx[0], idx[1]})")
        plt.xlabel("t")
        plt.ylabel("Amplitude")
        plt.ylim(output.min(), output.max())
        plt.xlim(0, nt * dt)

    iter = bool_slice(*output.shape, strides=[5, 5, 1], none_dims=[-1])
    frames = get_frames_bool(data=output, iter=iter, plotter=plotter)
    save_frames(frames, path='t')

    print('\n\nt.gif\n\n')


if __name__ == "__main__":
    main_big()
