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


def get_simple_field(field_size: float, n_cells: int, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_points = n_cells + 1
    step_size = grid_points * 1j  # funky mgrid convention to get step size
    mx, my = np.mgrid[:field_size:step_size, :field_size:step_size]
    z = amplitude * (np.sin(mx * 4 * np.pi) + np.cos(my * 4 * np.pi))
    return mx, my, z
