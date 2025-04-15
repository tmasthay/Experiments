import numpy as np
from scipy.optimize import minimize

# Define the function we want to minimize
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Initial guess for the variables
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

# Use the Nelder-Mead method
result = minimize(rosenbrock, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'disp': True})

# Display the result
print("Optimization Result:", result)
print("Optimal Parameters:", result.x)
print("Function Minimum:", result.fun)
