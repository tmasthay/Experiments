import copy
import numpy as np

def add_noise(data, noise_level):
    return data + np.random.normal(0.0, noise_level, data.shape)
    