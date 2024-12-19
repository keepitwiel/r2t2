from time import time

import numpy as np
from numba import njit
import matplotlib.pyplot as plt


EPSILON = 1e-6


@njit
def max_2x2_subdivisions(array: np.ndarray) -> np.ndarray:
    n = array.shape[0]  # Assuming matrix is n x n, where n = 2^p
    result = np.zeros((n // 2, n // 2), dtype=float)  # Result matrix of size (n//2) x (n//2)

    # Iterate through the matrix in steps of 2 for both rows and columns
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            # Get the 2x2 block and compute the maximum
            result[i // 2][j // 2] = np.max(array[i:i+2, j:j+2])

    return result


@njit
def get_max_arrays(array):
    h, w = array.shape
    assert h == w, f"Height and width should be the same. Found: height {h}, width {w}."
    p = int(np.log2(h))
    assert 2**p == h, f"Array dimension should be equal to 2**p, where p an integer >= 0. Found: {h}"
    result = [array]
    for i in range(1, p + 1):
        a = max_2x2_subdivisions(result[i - 1])
        result.append(a)
    return result


@njit
def max_test(max_arrays: list[np.ndarray], x: float, y: float, z: float) -> tuple[float, int]:
    for k in range(len(max_arrays) - 1, -1, -1):
        q = 2**k
        j, i = int(y // q), int(x // q)
        z_test = max_arrays[k][j, i]
        if z_test <= z:
            break
    return z_test, q


@njit
def get_partial_length(delta: float, position: float, step: int) -> float:
    index = position // step
    if delta < 0:
        if position % step == 0:
            minimum = (index - 1) * step
        else:
            minimum = index * step
        return (position - minimum) / -delta
    elif delta > 0:
        maximum = (index + 1) * step
        return (maximum - position) / delta
    elif delta == 0:
        return np.inf


@njit
def get_ray_length(max_arrays: list[np.ndarray], x: float, y: float, z: float, dx: float, dy: float) -> float:
    z_test, q = max_test(max_arrays, x, y, z)
    if z_test > z and q == 1:
        # even at the smallest resolution, the maximum value is above z.
        # therefore we conclude the ray is stopped.
        return 0.0
    lx = get_partial_length(dx, x, q)
    ly = get_partial_length(dy, y, q)
    l = min(lx, ly)
    l += EPSILON
    return l


@njit
def trace(max_arrays: list[np.ndarray], output: np.ndarray, j: int, i: int, dx: float, dy: float, dz: float, luminance: float) -> None:
    h, w = max_arrays[0].shape
    x = i + np.random.uniform()
    y = j + np.random.uniform()
    z =  max_arrays[0][j, i]
    while 0 < y < h and 0 < x < w:
        l = get_ray_length(max_arrays, x, y, z, dx, dy)
        if l == 0.0:
            luminance = 0.0
            break
        else:
            x += l * dx
            y += l * dy
            z += l * dz
    output[j, i] += luminance


@njit
def _render(max_arrays, output, cell_size, sun_azimuth, sun_altitude, n_sun_samples, n_sky_samples):
    h, w = max_arrays[0].shape
    dx = np.cos(sun_azimuth)
    dy = np.sin(sun_azimuth)
    dz = np.tan(sun_altitude) * cell_size
    for j in range(h):
        for i in range(w):
            for _ in range(n_sun_samples):
                trace(max_arrays, output, j, i, dx, dy, dz, 1.0)
            for _ in range(n_sky_samples):
                dx_ = np.random.uniform(-1, 1)
                dy_ = np.random.uniform(-1, 1)
                dz_ = np.random.uniform(0, cell_size)
                trace(max_arrays, output, j, i, dx_, dy_, dz_, 0.1)


class HeightMapPathTracer:
    def __init__(self, height_map, cell_size, sun_azimuth, sun_altitude, n_sun_samples, sun_luminosity, n_sky_samples, sky_luminosity):
        h, w = height_map.shape
        assert h == w, f"Height and width should be the same. Found: height {h}, width {w}."
        p = int(np.log2(h))
        assert 2 ** p == h, f"Array dimension should be equal to 2**p, where p an integer >= 0. Found: {h}"
        self.cell_size = cell_size
        self.sun_azimuth = sun_azimuth
        self.sun_altitude = sun_altitude
        self.n_sun_samples = n_sun_samples
        self.sun_luminosity = sun_luminosity
        self.n_sky_samples = n_sky_samples
        self.sky_luminosity = sky_luminosity
        self.max_arrays = get_max_arrays(height_map)
        self.illumination_map = np.zeros((h, w), dtype=float)

    def render(self):
        _render(
            self.max_arrays,
            self.illumination_map,
            self.cell_size,
            self.sun_azimuth,
            self.sun_altitude,
            self.n_sun_samples,
            self.n_sky_samples,
        )


DEFAULT_CELL_SIZE = 10.0
DEFAULT_AZIMUTH = -np.pi / 4
DEFAULT_ALTITUDE = np.pi / 4
DEFAULT_MAGNITUDE = 8


def main():
    d = 2**DEFAULT_MAGNITUDE
    my, mx = np.mgrid[-1:1:(d * 1j), -1:1:(d * 1j)]
    z = np.cos(my * 2 * np.pi) + np.sin(mx * 2 * np.pi)
    z += 1 - my**2 - mx**2
    z *= d

    tracer = HeightMapPathTracer(
        z,
        DEFAULT_CELL_SIZE,
        DEFAULT_AZIMUTH,
        DEFAULT_ALTITUDE,
        10,
        1.0,
        10,
        0.1
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(z, cmap="terrain")

    for _ in range(10):
        tracer.render()

    t0 = time()
    tracer.render()
    t1 = time()

    axes[1].set_title(f"render time: {t1 - t0:.3f}s")
    axes[1].imshow(tracer.illumination_map**(1/2.2), cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()
