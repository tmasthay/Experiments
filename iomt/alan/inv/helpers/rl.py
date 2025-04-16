import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import functools

class RiemannLiouvilleFractionalIntegral(nn.Module):
    def __init__(self, alpha, dt, max_length):
        """
        Compute the Riemann–Liouville fractional integral of order alpha,
        defined as (I^α f)(t) = 1/Γ(α) ∫₀ᵗ (t-τ)^(α-1) f(τ) dτ.

        Parameters:
            alpha: a scalar (float or 0D torch.Tensor) specifying the order.
                   Must be scalar-like (asserted).
            dt: time sampling interval.
            max_length: maximum length of the input signal (for precomputing the kernel).
        """
        super().__init__()
        
        # Ensure alpha is scalar-like.
        if isinstance(alpha, torch.Tensor):
            assert alpha.ndim == 0, "alpha must be a scalar (0D torch.Tensor)"
            self.alpha = float(alpha.item())
        else:
            self.alpha = float(alpha)
        
        self.dt = dt
        self.max_length = max_length
        
        # Precompute and store only the flipped kernel (1D) for efficiency.
        # Continuous kernel: h(t) = t^(alpha-1)/Γ(α); discretized:
        # h[k] = (dt^α * k^(α-1)) / Γ(α) for k = 0,1,...,max_length-1.
        # We set h[0] = 0 for stability.
        # Then flip it so that it is ready for use in a causal convolution.
        self._kernel_flip = self._precompute_kernel_flip(max_length)
    
    def _precompute_kernel_flip(self, L):
        k = torch.arange(L, dtype=torch.float32)
        kernel = (self.dt ** self.alpha) * (k.float() ** (self.alpha - 1)) / math.gamma(self.alpha)
        kernel[0] = 0.0  # handle singularity at k=0
        # Flip the kernel once here.
        kernel_flip = torch.flip(kernel, dims=[0])
        # Store as a 1D tensor.
        return kernel_flip  # shape: [L]

    @staticmethod
    def _prepare_kernel(kernel_flip, T):
        # Given a 1D kernel_flip of length L, extract the first T samples and reshape to [1, 1, T].
        return kernel_flip[:T].view(1, 1, T)
    
    def forward(self, f):
        """
        Compute the fractional integral of f.
        
        Parameters:
            f: a torch.Tensor with len(f.shape) >= 2.
               The last dimension (dim=-1) is the time dimension on which the integral is computed.
        
        Returns:
            A tensor of the same shape as f with the fractional integral applied along the last dimension.
        """
        # Assert that we have a batch dimension at least.
        assert isinstance(f, torch.Tensor) and len(f.shape) >= 2, "Input f must be a torch tensor with at least 2 dimensions."
        T = f.size(-1)
        assert T <= self.max_length, "Input length exceeds precomputed kernel length."
        
        # Prepare kernel: unsqueeze the 1D kernel_flip to [1, 1, T].
        kernel = self._prepare_kernel(self._kernel_flip.to(f.device), T)
        
        # Pad f on the left with (T - 1) zeros along the last dimension.
        f_padded = F.pad(f, (T - 1, 0))
        # Use conv1d along the last dimension; f is assumed to be batched.
        # Here f must be at least 2D, with shape [..., T]. We treat the last dimension as the temporal one.
        # We need to add a channel dimension: so assume input shape [B, T]; our conv1d expects [B, C, T].
        # Since our f has arbitrary batch dimensions, we flatten all but the last dimension.
        original_shape = f.shape  # e.g., (B1, B2, ..., T)
        f_flat = f.view(-1, T).unsqueeze(1)  # shape: [N, 1, T]
        f_padded_flat = F.pad(f_flat, (T - 1, 0))  # shape: [N, 1, T + T - 1]
        
        y_flat = F.conv1d(f_padded_flat, kernel)  # shape: [N, 1, T]
        y = y_flat.squeeze(1).view(*original_shape)
        return y

if __name__ == "__main__":

    # Test the fractional integral module.
    alpha = 0.8
    dt = 0.01
    T = 1000
    max_length = T

    frac_int = RiemannLiouvilleFractionalIntegral(alpha, dt, max_length)
    t = torch.linspace(0, (T - 1)*dt, T)
    f = torch.sin(2 * math.pi * t)
    # Add a batch dimension.
    f_batch = f.unsqueeze(0)  # shape: (1, T)
    y = frac_int(f_batch)
    print("Fractional integral output shape:", y.shape)