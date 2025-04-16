import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BatchedRiemannLiouvilleFractionalIntegralQuadrature(nn.Module):
    def __init__(self, alphas, dt, max_length):
        """
        Compute batched fractional integrals using integrated quadrature
        for each alpha in alphas. The fractional integral is given by:
            (I^α f)(t_n) = 1/Γ(α) ∫₀^(t_n) (t_n - τ)^(α-1) f(τ) dτ.
        
        We approximate the integral on each subinterval [t_j, t_{j+1}] by:
            (f_j + f_{j+1})/2 * ∫_(t_j)^(t_{j+1}) (t_n - τ)^(α-1) dτ,
        where the exact subinterval integral is:
            (( (n-j)*dt )^α - ((n-j-1)*dt)^α)/α.
        
        For alpha=0 we define the operator to be the identity.
        
        Parameters:
            alphas: 1D list or torch.Tensor of scalar orders (shape [num_alphas]).
            dt: time sampling interval.
            max_length: maximum number of time samples (for precomputing the kernel).
        """
        super().__init__()
        # Convert alphas to 1D tensor.
        if not isinstance(alphas, torch.Tensor):
            alphas = torch.tensor(alphas, dtype=torch.float32)
        assert alphas.ndim == 1, "alphas must be a 1D tensor."
        self.alphas = alphas
        self.num_alphas = alphas.numel()
        self.dt = dt
        self.max_length = max_length
        
        # Precompute integrated (and flipped) kernel for each alpha.
        # The kernel (for k>=1) is defined as:
        #    h[k] = ( ((k*dt)^α - ((k-1)*dt)^α)/α ) / Γ(α)
        # and we set h[0]=0.
        kernels = []
        for a in self.alphas:
            a_val = float(a.item())
            if abs(a_val) < 1e-8:
                # For alpha=0, the operator is the identity.
                # Represent this with a delta: h[0]=1, rest 0. (Note: length becomes max_length-1.)
                k = torch.zeros(max_length - 1, dtype=torch.float32)
                k[0] = 1.0
            else:
                k_range = torch.arange(max_length - 1, dtype=torch.float32)  # indices 0,..., max_length-1
                # Compute weights for k>=1.
                k_weight = torch.zeros_like(k_range)
                k_weight[0] = 0.0
                k_weight[1:] = (((k_range[1:] * dt) ** a_val - (((k_range[1:] - 1) * dt) ** a_val)) / a_val) / math.gamma(a_val)
                k = k_weight
            # Flip kernel along the time axis for conv1d.
            k_flip = torch.flip(k, dims=[0])
            kernels.append(k_flip)
        # Stack into tensor of shape [num_alphas, max_length-1]
        self.register_buffer("_kernel_flip", torch.stack(kernels, dim=0))
    
    def _prepare_kernel(self, idx, L):
        # Extract the kernel for alpha at index idx, taking the first L samples.
        # Shape: [1, 1, L]
        return self._kernel_flip[idx, :L].view(1, 1, L)
    
    def forward(self, f):
        """
        Compute the fractional integrals of f for each alpha using the improved quadrature.
        
        Parameters:
            f: a torch.Tensor with shape [..., T], with T time samples.
                 (Assert that len(f.shape) >= 2.)
        
        Returns:
            A tensor with shape [..., num_alphas, T-1], i.e. the same batch dimensions as f,
            an extra alpha dimension, and then the time dimension corresponding to the integrated values.
            (Note that this quadrature produces one fewer sample than the input.)
        """
        assert isinstance(f, torch.Tensor) and len(f.shape) >= 2, \
            "Input f must be a torch tensor with at least 2 dimensions."
        T = f.size(-1)
        assert T <= self.max_length, "Input length exceeds precomputed kernel length."
        
        original_shape = f.shape  # e.g., (..., T)
        # Form the averaged signal: average over adjacent samples.
        # Flatten all batch dims except time.
        f_flat = f.view(-1, T)  # shape: [N, T]
        # Compute averages over adjacent intervals: g_j = (f_j + f_{j+1})/2.
        g = (f_flat[:, :-1] + f_flat[:, 1:]) / 2.0  # shape: [N, T-1]
        # Add channel dimension: [N, 1, T-1].
        g = g.unsqueeze(1)
        
        outputs = []
        for i in range(self.num_alphas):
            # Prepare the kernel for current alpha; expected length is T-1.
            kernel = self._prepare_kernel(i, T - 1).to(f.device)  # shape: [1, 1, T-1]
            # To keep output length same as g, no extra padding is needed if we use 'valid' conv1d.
            y = F.conv1d(g, kernel)  # shape: [N, 1, T-1]
            outputs.append(y)
        # Stack along a new dimension: shape: [num_alphas, N, 1, T-1].
        out = torch.stack(outputs, dim=0).squeeze(2)  # shape: [num_alphas, N, T-1]
        # Permute to shape: [N, num_alphas, T-1].
        out = out.permute(1, 0, 2)
        # Reshape back to original batch dims with an extra alpha dimension.
        batch_shape = original_shape[:-1]
        input(out.shape)
        input(batch_shape)
        input(self.num_alphas)
        input(T-1)
        return out.view(*batch_shape, self.num_alphas, T - 1)

# Test the new implementation and plot for a sanity check.
if __name__ == "__main__":
    dt = 0.01
    T = 1000
    max_length = T
    # Test for alphas = [0, 1]:
    # For alpha=0, output should be the identity (i.e., f(t) itself, modulo one fewer sample).
    # For alpha=1, the classical integral: dt*cumsum(f).
    alphas = [0.0, 1.0]
    
    frac_int_quad = BatchedRiemannLiouvilleFractionalIntegralQuadrature(alphas, dt, max_length)
    
    # Create an input signal: e.g., a sine wave.
    t = torch.linspace(0, (T - 1) * dt, T)
    f = torch.sin(2 * math.pi * t)
    f_batch = f.unsqueeze(0)  # shape: [1, T]
    
    y = frac_int_quad(f_batch)  # shape: [1, num_alphas, T-1]
    y = y.squeeze(0)           # shape: [num_alphas, T-1]
    
    # Analytic classical integral for alpha = 1: dt * cumsum(f) (drop first sample for comparison).
    y1_analytic = torch.cumsum(f, dim=0) * dt
    y1_analytic = y1_analytic[1:]  # shape: [T-1]
    
    plt.figure(figsize=(10,6))
    plt.plot(t[1:].numpy(), f[1:].numpy(), 'r--', label="Input f(t)")
    plt.plot(t[1:].numpy(), y[0].detach().numpy(), label="Output for alpha=0 (Identity)")
    plt.plot(t[1:].numpy(), y[1].detach().numpy(), label="Output for alpha=1 (Numerical Integral)")
    plt.plot(t[1:].numpy(), y1_analytic.numpy(), 'k--', label="Analytic Integral (dt*cumsum)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Batched Riemann–Liouville Fractional Integral (Quadrature) Test")
    plt.legend()
    plt.savefig("batched_fractional_integral_quadrature_test.jpg")
    print("Plot saved as 'batched_fractional_integral_quadrature_test.jpg'")