import math
import torch
import matplotlib.pyplot as plt


class RiemannLiouvilleFractionalIntegral:
    def __init__(self, *, alpha: float, dt: float, max_length: int):
        assert 0.0 <= alpha <= 1.0, "We only handle 0≤α≤1 here"
        assert dt > 0.0, "dt must be positive"
        assert max_length > 0, "max_length must be positive"
        self.alpha = alpha
        self.dt = dt
        # prefactor Δt^α / Γ(α+1)
        prefac = dt**alpha / math.gamma(alpha + 1.0)
        # build the w_k = k^α - (k-1)^α kernel
        k = torch.arange(max_length, dtype=torch.float32)
        w = torch.zeros_like(k)
        w[1:] = k[1:]**alpha - k[:-1]**alpha
        # scale once and for all
        self.weights = w * prefac  # shape [max_length]

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: 1D tensor length N ≤ max_length
        returns   I^α f at each n:   sum_{k=0}^n w_{k+1} * f[n-k]
        """
        if self.alpha == 0.0:
            return f
        N = f.shape[0]
        # move precomputed kernel onto the right device & dtype
        w = self.weights.to(f.device).to(f.dtype)
        out = torch.zeros_like(f)
        for n in range(N-1):
            # w[1]…w[n+1] paired with f[n]…f[0]
            wn = w[1 : n+2]               # length n+1
            fn = f[: n+1].flip(0)        # length n+1
            # input(f'{N=}, {n=}, {wn.shape=}, {fn.shape=}')
            out[n] = torch.dot(wn, fn)
        out[-1] = out[-2]
        return out

if __name__ == "__main__":
    dt = 0.1
    N = 200
    t = torch.arange(N, dtype=torch.float32) * dt

    # two test signals
    f_const = torch.ones(N, dtype=torch.float32)
    f_sin   = torch.sin(2 * math.pi * t)

    alphas = [0.0, 0.02, 0.5, 1.0]

    plt.figure(figsize=(12, 4))

    # constant input
    plt.subplot(1, 2, 1)
    for α in alphas:
        rl = RiemannLiouvilleFractionalIntegral(α, dt, N)
        out = rl(f_const)
        plt.plot(t.numpy(), out.numpy(), label=f"α={α}")
    plt.title("I^α[1]")
    plt.xlabel("t")
    plt.legend()

    # sine input
    plt.subplot(1, 2, 2)
    for α in alphas:
        rl = RiemannLiouvilleFractionalIntegral(α, dt, N)
        out = rl(f_sin)
        plt.plot(t.numpy(), out.numpy(), label=f"α={α}")
    plt.title("I^α[sin(2πt)]")
    plt.xlabel("t")
    plt.legend()

    plt.tight_layout()
    plt.savefig("rl_integral_refined.png", dpi=300)
    print("Saved plot to rl_integral_refined.png")