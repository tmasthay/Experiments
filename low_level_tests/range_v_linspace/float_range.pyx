# float_range_cython.pyx

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def float_range(double start, double stop, int num_points):
    cdef int i
    for i in range(num_points):
        yield start + i * (stop - start) / (num_points - 1)
