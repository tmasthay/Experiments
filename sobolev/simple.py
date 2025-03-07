# hydra_script.py

import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim

def hs_norm(f, s, dx=1.0):
    """
    Compute the Hilbert-Sobolev (Bessel potential) norm of a 1D torch.Tensor.
    """
    if s == 0:
        # L2 norm
        return torch.sum(f**2) * dx

    n = f.shape[0]
    freqs = torch.fft.fftfreq(n, d=dx).to(f.device)
    f_hat = torch.fft.fft(f)
    f_hat_sq = f_hat.real**2 + f_hat.imag**2
    kernel = (1.0 + freqs**2)**s
    return torch.sum(kernel * f_hat_sq) * (dx / n)

@hydra.main(config_path="all/simple", config_name="default")
def main(cfg: DictConfig):
    # Generate time samples
    t = torch.linspace(0, cfg.tmax, cfg.n, requires_grad=False)

    # True signal
    f = torch.sin(2.0 * 3.14159 * cfg.freq_true * t + cfg.phase_true)

    # Add Gaussian noise
    noise = torch.randn_like(f) * cfg.noise_std
    f_noisy = f + noise

    # Initialize frequency & phase as learnable parameters
    freq_guess = torch.nn.Parameter(torch.tensor(cfg.freq_guess_init))
    phase_guess = torch.nn.Parameter(torch.tensor(cfg.phase_guess_init))
    optimizer = optim.Adam([freq_guess, phase_guess], lr=cfg.lr)

    for step in range(cfg.num_steps):
        optimizer.zero_grad()
        guess = torch.sin(2.0 * 3.14159 * freq_guess * t + phase_guess)
        misfit = hs_norm(f_noisy - guess, cfg.s, dx=cfg.dx)
        misfit.backward()
        optimizer.step()

        if step % cfg.log_interval == 0:
            print(f"Step {step} | freq_guess={freq_guess.item():.3f}, "
                  f"phase_guess={phase_guess.item():.3f}, misfit={misfit.item():.6f}")

if __name__ == "__main__":
    main()
