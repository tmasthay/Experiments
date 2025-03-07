# hydra_script.py

import os
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import torch
import torch.optim as optim
from mh.core import hydra_out
from misfit_toys.utils import bool_slice
from mh.typlotlib import save_frames, get_frames_bool, bool_slice, setup_gg_plot
import random
import numpy as np
from misfit_toys.utils import git_dump_info
from os.path import join as pj

# setup_gg_plot(clr_in='black')
torch.manual_seed(507)


def hs_norm(f, *, s, beta=1.0, alpha=1.0, dx=1.0):
    """
    Compute the Hilbert-Sobolev (Bessel potential) norm of a 1D torch.Tensor.
    """
    if s == 0:
        return torch.sum(f**2) * dx

    n = f.shape[0]
    freqs = torch.fft.fftfreq(n, d=dx).to(f.device)
    f_hat = torch.fft.fft(f)
    f_hat_sq = f_hat.real**2 + f_hat.imag**2
    kernel = (alpha + beta * freqs**2) ** s
    return torch.sum(kernel * f_hat_sq) * (dx / n)


def plotter(*, data, idx, fig, axes, noisy_signal, ground_truth, t):
    plt.clf()
    plt.plot(t, noisy_signal, label="Noisy signal", color="red", linestyle="--")
    plt.plot(
        t, ground_truth, label="Ground truth", color="green", linestyle="--"
    )
    plt.plot(t, data[idx], label="Current guess", color="blue")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"Step {idx}")
    plt.tight_layout()


@hydra.main(config_path="all/simple", config_name="default", version_base=None)
def main(cfg: DictConfig):
    with open(pj(hydra_out('git_info.txt')), 'w') as f:
        f.write(git_dump_info())

    two_pi = 2.0 * 3.14159

    # Generate time samples
    t = torch.linspace(0, cfg.tmax, cfg.n, requires_grad=False)

    # True signal
    f = torch.sin(two_pi * cfg.freq_true * t + cfg.phase_true)

    # Add Gaussian noise
    noise = torch.randn_like(f) * cfg.noise_std
    f_noisy = f + noise

    # Create an initial guess (e.g., a sine wave with some arbitrary phase)
    init_guess = torch.sin(
        two_pi * cfg.freq_guess_init * t + cfg.phase_guess_init
    )

    # Make the entire guess a learnable parameter
    guess = torch.nn.Parameter(init_guess.clone())

    # Optimizer
    optimizer = optim.Adam([guess], lr=cfg.lr)

    res = [guess.detach().clone()]
    append_freq = max(1, cfg.num_steps // cfg.num_frames)
    for step in range(cfg.num_steps):
        optimizer.zero_grad()

        # Hilbert-Sobolev norm of residual
        beta = cfg.beta_over_alpha * cfg.alpha
        misfit = hs_norm(
            f_noisy - guess, s=cfg.s, alpha=cfg.alpha, beta=beta, dx=cfg.dx
        )
        misfit.backward()

        optimizer.step()

        if step % append_freq == 0:
            res.append(guess.detach().clone())

        if step % cfg.log_interval == 0:
            print(f"Step {step} | misfit={misfit.item():.6f}")

    res = torch.stack(res, dim=0)
    iter = bool_slice(*res.shape, none_dims=[1])
    frames = get_frames_bool(
        data=res,
        iter=iter,
        plotter=plotter,
        t=t,
        ground_truth=f,
        noisy_signal=f_noisy,
    )
    save_frames(frames, path=hydra_out("res.gif"))

    print(hydra_out("res.gif"))
    vs_code_exe = '/home/tyler/.vscode-server/cli/servers/Stable-cd4ee3b1c348a13bafd8f9ad8060705f6d4b9cba/server/bin/remote-cli/code'
    os.system(f"{vs_code_exe} {hydra_out('res.gif')}")


if __name__ == "__main__":
    main()
