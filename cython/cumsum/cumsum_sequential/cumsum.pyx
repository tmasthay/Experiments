# cumsum.pyx
# cython: language_level=3

import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t DTYPE_t

def cumsum(cnp.ndarray[DTYPE_t, ndim=1] x):
    cdef int n, N
    cdef cnp.ndarray[DTYPE_t, ndim=1] y

    n = 1
    N = len(x)
    y = np.empty(N, dtype=np.float32)

    while( n < N ):
        y[n] = y[n-1] + x[n]
        n += 1
    return y

