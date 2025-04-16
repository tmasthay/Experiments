import os
import torch
import deepwave as dw
import matplotlib.pyplot as plt
from mh.typlotlib import save_frames, get_frames_bool, bool_slice
from torch.nn import functional as F
from mh.core import DotDictImmutable as DDI
from rl_batch import BatchedRiemannLiouvilleFractionalIntegral as RLInt
import hydra
from omegaconf import DictConfig, OmegaConf
from mh.core import Tee

# -- Weight Scheduler Example --
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

# -- RL Loss using RLInt --
def rl_loss(observed_data, alphas, alpha_weights, dt, max_length, gamma=0.1):
    """
    Compute the Riemann-Liouville fractional integral loss for a batch of data.
    
    Returns a function that computes the loss given simulated data.
    """
    preprocess = torch.nn.Softplus(beta=1.0, threshold=20.0)
    rl_int = RLInt(alphas=alphas, dt=dt, max_length=max_length, gamma=gamma)
    filtered_obs = preprocess(observed_data)

    def helper(sim_data):
        # Compute fractional integrals of both simulated and observed data.
        filtered_sim_data = preprocess(sim_data)
        sim_data_rl = rl_int(filtered_sim_data - filtered_obs)
        # Multiply by scheduled weights
        res = alpha_weights()[:, None] * sim_data_rl
        return (sim_data_rl ** 2).mean()
    return helper

# -- Identity Moment Source Model --
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
        self.mu = torch.nn.Parameter(mu, requires_grad=mu.requires_grad).to(device)
        self.sig = torch.nn.Parameter(sig, requires_grad=sig.requires_grad).to(device)
        self.peak_time = torch.nn.Parameter(peak_time, requires_grad=peak_time.requires_grad).to(device)
        self.freq = torch.nn.Parameter(freq, requires_grad=freq.requires_grad).to(device)
        self.scale = torch.nn.Parameter(scale, requires_grad=scale.requires_grad).to(device)
        self.device = device
        self.num_shots = self.src_loc.shape[0]
        self.num_sources = self.src_loc.shape[1]
        self.nt = nt
        self.dt = dt
        self.target_shape = torch.Size([self.num_shots, self.num_sources, self.nt])
        self.__validate_parameters()

    def __validate_parameters(self):
        def check_elem_dim(t, dims, elems, name):
            assert t.ndim == dims, f"{name} must be {dims}D, got {t.ndim}D"
            assert t.nelement() == elems, f"{name} must have {elems} elements, got {t.nelement()}"
        assert self.src_loc.ndim == 3, f"src_loc must be 3D, got {self.src_loc.ndim}D"
        assert self.src_loc.shape[2] == 2, f"{self.src_loc.shape[1]=} != 2"
        assert self.src_loc.dtype == torch.int, f"src_loc must be int, got {self.src_loc.dtype}"
        check_elem_dim(self.mu, 1, 2, "mu")
        check_elem_dim(self.sig, 1, 2, "sig")
        check_elem_dim(self.peak_time, 0, 1, "peak_time")
        check_elem_dim(self.freq, 0, 1, "freq")
        check_elem_dim(self.scale, 0, 1, "scale")
        assert self.nt > 0, f"nt must be positive, got {self.nt}"
        assert self.dt > 0, f"dt must be positive, got {self.dt}"
        def check_device(t, name):
            assert torch.device(t.device) == torch.device(self.device), f"{name} must be on {self.device}, got {t.device}"
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
        arg = -torch.sum((self.src_loc.float() - self.mu[None, None, :]) / self.sig[None, None, :], dim=-1)
        exp_term = torch.exp(arg)
        res = wavelet[None, None, :] * exp_term[:, :, None] * self.scale
        assert res.shape == self.target_shape, f"Expected shape {self.target_shape}, got {res.shape}"
        return res.to(self.device)

# -- Utility Functions for History Logging --
def get_grad_clone(t):
    return None if t.grad is None else t.grad.clone().cpu().numpy()

def get_hist(t):
    data_clone = t.detach().clone().cpu().numpy()
    grad_clone = get_grad_clone(t)
    return {"data": data_clone, "grad": grad_clone}

# -- Main Training Function using Hydra --
@hydra.main(config_path="all/main", config_name="default", version_base=None)
@Tee.hydra_tee
def main(cfg: DictConfig):
    c = DDI(OmegaConf.to_container(cfg, resolve=True))
    device = c.device
    nt = c.nt
    dt = c.dt
    grid_spacing = c.grid_spacing
    nx = c.nx
    ny = c.ny

    vp = torch.ones((ny, nx), device=device) * c.vp

    # Process source location from config.
    src_loc = torch.tensor(c.src_loc).int().to(device)
    # Process receiver locations.
    if c.rec_locs.method == "cartesian":
        rec_locs = torch.cartesian_prod(
            torch.tensor([c.rec_locs.fixed_y]).int(), torch.arange(0, nx, c.rec_locs.x_stride).int()
        ).unsqueeze(0).to(device)
    else:
        raise ValueError("Unsupported rec_locs method.")

    # Ground-truth parameters.
    true_params = c.ground_truth
    true_mu = torch.tensor(true_params.mu).to(device)
    true_sig = torch.tensor(true_params.sig).to(device)
    true_peak_time = torch.tensor(true_params.peak_time).to(device)
    true_freq = torch.tensor(true_params.freq).to(device)
    true_scale = torch.tensor(true_params.scale).to(device)

    # Create ground-truth model.
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
    true_src_amp = source_true.forward()
    syn_data = dw.scalar(
        vp,
        grid_spacing,
        dt,
        source_locations=src_loc,
        receiver_locations=rec_locs,
        source_amplitudes=true_src_amp,
        pml_width=10,
    )[-1]
    # Add noise.
    rms = torch.sqrt(torch.mean(syn_data**2, dim=-1, keepdim=True))
    noise = torch.randn_like(syn_data) * 0.1 * rms
    syn_data = syn_data.detach() + noise

    # Loss function (RL loss).
    _loss = rl_loss(
        observed_data=syn_data,
        alphas=c.alphas,
        alpha_weights=PiecewiseAlphaScheduler(num_alphas=len(c.alphas), step_size=c.training.num_epochs//len(c.alphas), device=device),
        dt=dt,
        max_length=nt,
        gamma=c.gamma,
    )

    # Utility function to format messages.
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

    init_params = c.init
    init_mu = torch.tensor(init_params.mu).to(device).requires_grad_(True)
    init_sig = torch.tensor(init_params.sig).to(device).requires_grad_(True)
    init_peak_time = torch.tensor(init_params.peak_time).to(device).requires_grad_(True)
    init_freq = torch.tensor(init_params.freq).to(device).requires_grad_(True)
    init_scale = torch.tensor(init_params.scale).to(device).requires_grad_(True)

    init_msg = get_msg(init_mu, init_sig, init_peak_time, init_freq, init_scale, 'Initial guess:')
    truth_msg = get_msg(true_mu, true_sig, true_peak_time, true_freq, true_scale, 'Ground truth:')

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

    # Set up optimizer using Hydra config.
    if c.optim.name == "LBFGS":
        optimizer = torch.optim.LBFGS(source_model.parameters(), **c.optim.params)
    elif c.optim.name == "Adam":
        optimizer = torch.optim.Adam(source_model.parameters(), **c.optim.params)
    elif c.optim.name == "SGD":
        optimizer = torch.optim.SGD(source_model.parameters(), **c.optim.params)
    else:
        raise ValueError(f"Unknown optimizer: {c.optim.name}")

    abs_loss_tol = c.loss.abs_loss_tol
    num_epochs = c.training.num_epochs
    history_freq = c.training.history_freq
    history = []

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

        loss_val = optimizer.step(closure)
        curr_loss = loss_val.item()

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

if __name__ == "__main__":
    main()