import numpy as np
import math

def batched_riemann_liouville_fractional_integral(x, alphas, dt=1.0):
    """
    Compute the Riemann-Liouville fractional integral for multiple orders (alphas) on input signals.
    
    Parameters:
        x: np.ndarray of shape (..., T)
           Input signal(s) along the last dimension (time series data).
        alphas: array-like of float
           List or array of fractional integration orders to apply.
        dt: float, optional (default=1.0)
           Time step between consecutive samples in the input. Assumed constant.
           
    Returns:
        np.ndarray of shape (..., len(alphas), T)
        Fractional integrated signals for each alpha. The new axis for alpha is inserted 
        before the time dimension.
    """
    x = np.array(x, dtype=float)            # ensure input as NumPy array
    alphas = np.array(alphas, dtype=float)  # array of alpha values
    leading_shape = x.shape[:-1]            # shape of any leading (non-time) dimensions
    T = x.shape[-1]                        # number of time points
    num_alphas = alphas.size
    
    # Initialize output array with the desired shape: (*leading_shape, num_alphas, T)
    out = np.zeros(leading_shape + (num_alphas, T), dtype=float)
    
    for ai, alpha in enumerate(alphas):
        if alpha == 0.0:
            # 0th fractional integral is the identity (just copy the input signal)
            out[..., ai, :] = x
        else:
            # Compute 1/Gamma(alpha) once for efficiency
            inv_gamma = 1.0 / math.gamma(alpha)
            # Set initial time point result: integral from 0 to 0 is zero for alpha > 0
            out[..., ai, 0] = 0.0
            # Iterate through time steps, applying trapezoidal rule increment
            for n in range(1, T):
                f_n    = x[..., n]     # f at current time t_n
                f_prev = x[..., n-1]   # f at previous time t_{n-1}
                # Incremental fractional integral from t_{n-1} to t_n:
                # ΔI^α_n = (dt^α / ((α+1)*Γ(α))) * (f_n/α + f_prev)
                increment = ( (dt ** alpha) * inv_gamma / (alpha + 1) ) * (f_n / alpha + f_prev)
                out[..., ai, n] = out[..., ai, n-1] + increment
    return out

import numpy as np

# Create time array and sine wave input
t = np.linspace(0, 2*np.pi, 100)            # 0 to 2π with 100 points
x = np.sin(t)                              # f(t) = sin(t)
alphas = [0.0, 1.0]
dt = t[1] - t[0]                           # uniform time step
result = batched_riemann_liouville_fractional_integral(x, alphas, dt=dt)

print(result.shape)        # Expected shape: (2, 100) since we have 2 alphas and T=100
# Verify α=0 case (result[0] vs original x)
print("Max error for α=0:", np.max(np.abs(result[0] - x)))
# Verify α=1 case (result[1] vs exact 1-cos(t))
exact_integral = 1 - np.cos(t)
print("Max error for α=1:", np.max(np.abs(result[1] - exact_integral)))