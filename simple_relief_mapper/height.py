import numpy as np
from noise import snoise2


def simplex_height_map(dim, octaves, amplitude, seed=42):
    """
    Simple wrapper that samples a height map from simplex noise.
    :param dim:
    :param octaves:
    :param amplitude:
    :param seed:
    :return:
    """
    arr = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            arr[j, i] = snoise2(j / dim, i / dim, octaves=octaves, base=seed)
    arr *= amplitude
    return arr
