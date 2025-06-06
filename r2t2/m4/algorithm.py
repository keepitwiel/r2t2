import numpy as np

from .line import line
from .m4 import M4
from .algo_functions_parallel import fill_shadow_array


def algorithm(
    m4: M4,
    azimuth: float,
    altitude: float,
    cell_size: float,
) -> list[np.ndarray]:
    assert 0 <= altitude < np.pi / 2, "altitude should be between 0 and pi / 2."
    dimension = m4.get_dimension()
    n_levels = m4.get_n_levels()
    line_coordinates, projection_length = line(azimuth, dimension)
    result = [np.array([[False]])]
    k = 0
    global_max = m4.get_global_max()
    for level in range(n_levels - 1, -1, -1):
        step_size = 2 ** level
        previous_array = result[k]
        d = previous_array.shape[0]
        shadow_array = np.zeros((d * 2, d * 2), dtype=bool)
        min_array = m4.get_array(level=level, is_max=False)
        max_array = m4.get_array(level=level, is_max=True)
        dz = step_size * cell_size * projection_length * np.tan(altitude)
        fill_shadow_array(shadow_array, previous_array, min_array, max_array, line_coordinates, dz, global_max)
        result.append(shadow_array)
        k += 1
    return result