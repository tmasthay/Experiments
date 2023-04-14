
def float_range_pure(start, stop, num_points):
    for i in range(num_points):
        yield start + i * (stop - start) / (num_points - 1)
