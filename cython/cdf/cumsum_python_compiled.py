import numpy as np
def cumsum(x):
    y = np.zeros(len(x) + 1)
    for i in range(1, len(x)): 
        y[i] = y[i-1] + x[i]
    return y