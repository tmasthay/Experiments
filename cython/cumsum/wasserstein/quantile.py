import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import accumulate
import os

def quantile(p, U, dt, ot, div_tol=1e-16, tail_tol=1e-30):
    Q = np.zeros(len(p))
    q = 0
    t = np.where(U > 0)[0][0]
    for pp in p:
        if( t >= len(U) - 1 or pp > 1-tail_tol ):
            Q[q] = ot + (len(U) - 1) * dt
        elif( U[t] == 0.0 ): Q[q] = ot
        else:
            while( t < len(U) - 1 and U[t+1] < pp ): t += 1
            if( t == len(U) - 1 ): Q[q] = (len(U) - 1)*dt
            else:
                if( abs(U[t+1] - U[t]) >= div_tol ):
                    Q[q] = ot + dt * (t + (pp-U[t]) / (U[t+1] - U[t]))
                else:
                    Q[q] = ot + dt * t
        q += 1
    return Q
