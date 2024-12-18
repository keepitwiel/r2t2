import numpy as np
import matplotlib.pyplot as plt
from numba import njit


CELL_SIZE = 10.0
DEFAULT_AZIMUTH = np.pi / 3
INCLINATION = np.pi / 4
MAGNITUDE = 6
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
def trace(max_arrays: list[np.ndarray], z: np.ndarray, s: np.ndarray, j: int, i: int, dx: float, dy: float, dz: float) -> None:
    h, w = z.shape
    x = i + np.random.uniform()
    y = j + np.random.uniform()
    z_ = z[j, i]
    value = 1.0
    while 0 < y < h and 0 < x < w:
        l = get_ray_length(max_arrays, x, y, z_, dx, dy)
        if l == 0.0:
            value = 0.0
            break
        else:
            x += l * dx
            y += l * dy
            z_ += l * dz
    s[j, i] += value

@njit
def raycast(max_arrays, z, output, azimuth=DEFAULT_AZIMUTH):
    h, w = z.shape
    if 0 <= INCLINATION < np.pi / 2:
        dx = np.cos(azimuth)
        dy = np.sin(azimuth)
        dz = np.tan(INCLINATION) * CELL_SIZE
        for j in range(h):
            for i in range(w):
                for _ in range(10):
                    trace(max_arrays, z, output, j, i, dx, dy, dz)


def main():
    d = 2**MAGNITUDE
    my, mx = np.mgrid[-1:1:(d * 1j), -1:1:(d * 1j)]
    z = np.cos(my * 2 * np.pi) + np.sin(mx * 2 * np.pi)
    z += 1 - my**2 - mx**2
    z *= d

    max_arrays = get_max_arrays(z)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].imshow(z, cmap="terrain")

    for azimuth in np.linspace(0, 2 * np.pi, 361):
        plt.cla()
        output = np.zeros(z.shape, dtype=float)
        raycast(max_arrays, z, output, azimuth=azimuth)
        axes[0].set_title(azimuth)
        axes[1].imshow(output)
        # plt.show()
        plt.pause(0.0001)


if __name__ == "__main__":
    main()
