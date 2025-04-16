import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BatchedRiemannLiouvilleFractionalIntegral(nn.Module):
    def __init__(self, alphas, dt, max_length):
        """
        Compute batched Riemann–Liouville fractional integrals for each alpha in alphas,
        defined (formally) as:
            (I^α f)(t) = 1/Γ(α) ∫₀ᵗ (t-τ)^(α-1) f(τ) dτ.
        
        For discrete approximations we precompute a kernel for each alpha.
        For alpha=0, we define the kernel to be a delta impulse (i.e. the identity).
        
        Parameters:
            alphas: 1D list or torch.Tensor of scalars (shape [num_alphas]).
            dt: time sampling interval.
            max_length: maximum length of the input signal for precomputing the kernels.
        """
        super().__init__()
        # Convert alphas to a tensor and ensure it is 1D.
        if not isinstance(alphas, torch.Tensor):
            alphas = torch.tensor(alphas, dtype=torch.float32)
        assert alphas.ndim == 1, "alphas must be a 1D tensor."
        self.alphas = alphas  # shape: [num_alphas]
        self.num_alphas = alphas.numel()
        self.dt = dt
        self.max_length = max_length
        
        # Precompute the flipped kernel for each alpha.
        kernels = []
        for a in self.alphas:
            a_val = float(a.item())
            if abs(a_val) < 1e-8:  # Special case: alpha == 0 => identity operator.
                # Delta: impulse at t = 0.
                k = torch.zeros(max_length, dtype=torch.float32)
                k[0] = 1.0
            else:
                k_vals = torch.arange(max_length, dtype=torch.float32)
                # Discretized kernel: h[k] = (dt^α * k^(α-1)) / Gamma(α), with h[0]=0.
                k_vals = (self.dt ** a_val) * (k_vals.float() ** (a_val - 1)) / math.gamma(a_val)
                k_vals[0] = 0.0  # avoid singularity at 0
                k = k_vals
            # Flip the kernel so that conv1d (which does cross-correlation) computes the correct convolution.
            k_flip = torch.flip(k, dims=[0])
            kernels.append(k_flip)
        # Stack into a tensor of shape [num_alphas, max_length]
        self.register_buffer("_kernel_flip", torch.stack(kernels, dim=0))
    
    def _prepare_kernel(self, idx, T):
        # Extract and reshape the kernel for alpha at index idx.
        # Final shape will be [1, 1, T] for use in conv1d.
        return self._kernel_flip[idx, :T].view(1, 1, T)
    
    def forward(self, f):
        """
        Compute the fractional integrals of f for each alpha.
        
        Parameters:
            f: a torch.Tensor with shape [..., T], where T is the time dimension.
               (An assert will ensure len(f.shape) >= 2.)
        
        Returns:
            A tensor with shape [..., num_alphas, T], i.e. the same batch dimensions as f,
            an extra dimension for the different alphas, and then the time dimension.
        """
        assert isinstance(f, torch.Tensor) and len(f.shape) >= 2, \
            "Input f must be a torch tensor with at least 2 dimensions."
        T = f.size(-1)
        assert T <= self.max_length, "Input length exceeds precomputed kernel length."
        
        original_shape = f.shape  # e.g. (B1, B2, ..., T)
        # Flatten all batch dimensions except time.
        f_flat = f.view(-1, T)  # shape: [N, T]
        f_flat = f_flat.unsqueeze(1)  # shape: [N, 1, T]
        # Pad on the left with (T - 1) zeros to ensure causality.
        f_padded = F.pad(f_flat, (T - 1, 0))  # shape: [N, 1, 2T - 1]
        
        outputs = []
        for i in range(self.num_alphas):
            kernel = self._prepare_kernel(i, T).to(f.device)  # shape: [1, 1, T]
            y = F.conv1d(f_padded, kernel)  # shape: [N, 1, T]
            outputs.append(y)
        # Stack outputs along new dimension: shape [num_alphas, N, 1, T]
        out = torch.stack(outputs, dim=0).squeeze(2)  # shape: [num_alphas, N, T]
        # Permute to shape [N, num_alphas, T]
        out = out.permute(1, 0, 2)
        # Reshape back to original batch dimensions with the extra alpha dimension inserted.
        batch_shape = original_shape[:-1]
        return out.view(*batch_shape, self.num_alphas, T)

if __name__ == "__main__":
    # Set test parameters.
    dt = 0.01
    T = 1000
    max_length = T
    # Test for alphas = [0, 1]:
    #   alpha = 0 should act as the identity (output equals input)
    #   alpha = 1 should give the classical integral, i.e. approximate cumulative sum times dt.
    # alphas = [0.0, 0.5, 1.0]
    # alphas = torch.linspace(0, 1, 10)
    epsilon = 0.1
    alphas = [0.0, epsilon, 1-epsilon, 1.0]
    
    # Instantiate the batched fractional integral operator.
    frac_int = BatchedRiemannLiouvilleFractionalIntegral(alphas, dt, max_length)
    
    # Create an input signal, e.g., a sine wave.
    t = torch.linspace(0, (T - 1) * dt, T)
    f = torch.sin(2 * math.pi * t)
    # Add a batch dimension: shape [1, T]
    f_batch = f.unsqueeze(0)
    
    # Compute the batched fractional integral.
    y = frac_int(f_batch)  # shape: [1, num_alphas, T]
    y = y.squeeze(0)       # shape: [num_alphas, T]
    
    # For sanity:
    # For alpha = 0, output should match f.
    # For alpha = 1, we compare to the classical cumulative integral: dt * cumsum(f)
    y1_analytic = torch.cumsum(f, dim=0) * dt
    
    # Plot results.
    plt.figure(figsize=(10,6))
    plt.plot(t.numpy(), y[0].detach().numpy(), label="Output for alpha=0 (Identity)")
    plt.plot(t.numpy(), f.numpy(), 'r--', label="Input f(t)")
    plt.plot(t.numpy(), y[-1].detach().numpy(), label="Output for alpha=1 (Numerical Integral)")
    plt.plot(t.numpy(), y1_analytic.numpy(), 'k--', label="Analytic Integral (dt*cumsum)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Batched Riemann–Liouville Fractional Integral Test")
    
    for i in range(1, len(alphas)-1):
        plt.plot(t.numpy(), y[i].detach().numpy(), label=f"Output for alpha={alphas[i]}")
        
    plt.legend()
    plt.savefig("batched_fractional_integral_test.jpg")
    print("Plot saved as 'batched_fractional_integral_test.jpg'")