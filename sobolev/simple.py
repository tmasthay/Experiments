import torch

def hs_norm(f, s, dx=1.0):
    """
    f  : 1D torch.Tensor of real (or complex) samples
    s  : float, the fractional order
    dx : float, spacing in the 'time' (or spatial) domain
    
    Returns the Bessel-potential-based Hilbert-Sobolev norm, backprop-ready.
    """
    # If s=0, just return the L2 norm
    if s == 0:
        return torch.sum(f**2) * dx

    n = f.shape[0]
    # Frequencies for discrete FFT
    freqs = torch.fft.fftfreq(n, d=dx).to(f.device)

    # Compute the FFT
    f_hat = torch.fft.fft(f)

    # Modulus squared of the FFT
    f_hat_sq = f_hat.real**2 + f_hat.imag**2

    # (1 + |freq|^2)^s factor
    kernel = (1.0 + freqs**2)**s

    # Discrete version of the integral
    # Multiply by dx/n to approximate continuous norm scaling
    return torch.sum(kernel * f_hat_sq) * (dx / n)
