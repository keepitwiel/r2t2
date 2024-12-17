import numpy as np
import matplotlib.pyplot as plt
from numba import njit


CELL_SIZE = 10.0
DEFAULT_AZIMUTH = np.pi / 3
INCLINATION = np.pi / 4
MAGNITUDE = 7


def max_2x2_subdivisions(matrix):
    n = matrix.shape[0]  # Assuming matrix is n x n, where n = 2^p
    result = np.zeros((n // 2, n // 2), dtype=float)  # Result matrix of size (n//2) x (n//2)

    # Iterate through the matrix in steps of 2 for both rows and columns
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            # Get the 2x2 block and compute the maximum
            result[i // 2][j // 2] = np.max(matrix[i:i+2, j:j+2])

    return result


class SimpleQuadTree:
    def __init__(self, array):
        h, w = array.shape
        assert h == w, f"Height and width should be the same. Found: height {h}, width {w}."
        p = int(np.log2(h))
        assert 2**p == h, f"Array dimension should be equal to 2**p, where p an integer >= 0. Found: {h}"
        self.p = p
        self.max_arrays = self.get_max_arrays(array)

    def get_max_arrays(self, array):
        result = [array]
        for i in range(1, self.p + 1):
            a = max_2x2_subdivisions(result[i - 1])
            result.append(a)
        return result

    def foo(self, x, y, z):
        for k in range(self.p, -1, -1):
            q = 2 ** k
            j, i = int(y // q), int(x // q)
            z_test = self.max_arrays[k][j, i]
            if z_test <= z:
                break
        return z_test, q

def get_ray_length(tree, x, y, z, dx, dy) -> float:
    """
    Get the ray length l such that ray at x + l * dx, y + l * dy, z + l * dz needs to be tested again
    :param x:
    :param y:
    :param z:
    :param dx:
    :param dy:
    :return: True/False if ray is stopped, length
    """
    z_test, q = tree.foo(x, y, z)

    if z_test > z and q == 1:
        # even at the smallest resolution, the maximum value is above z.
        # therefore we conclude the ray is stopped.
        return 0.0

    j, i = y // q, x // q

    if dy < 0 and y % q == 0:
        ymin = (j - 1) * q
    else:
        ymin = j * q
    ymax = ymin + q

    if dx < 0 and x % q == 0:
        xmin = (i - 1) * q
    else:
        xmin = i * q
    xmax = xmin + q

    if dx < 0:
        lx = (x - xmin) / -dx
    elif dx > 0:
        lx = (xmax - x) / dx
    elif dx == 0:
        lx = np.inf

    if dy < 0:
        ly = (y - ymin) / -dy
    elif dy > 0:
        ly = (ymax - y) / dy
    elif dy == 0:
        ly = np.inf

    l = min(lx, ly)
    return l

def trace(tree, z, s, j, i, dx, dy, dz):
    h, w = z.shape
    x = i + 0.5
    y = j + 0.5
    z_ = z[j, i]
    while 0 < y < h and 0 < x < w:
        l = get_ray_length(tree, x, y, z_, dx, dy)
        if l == 0.0:
            s[j, i] = False
            break
        else:
            x += l * dx
            y += l * dy
            z_ += l * dz


def raycast(tree, z, s, azimuth=DEFAULT_AZIMUTH):
    h, w = z.shape
    if 0 <= INCLINATION < np.pi / 2:
        dx = np.cos(azimuth)
        dy = np.sin(azimuth)
        dz = np.tan(INCLINATION) * CELL_SIZE
        for j in range(h):
            for i in range(w):
                trace(tree, z, s, j, i, dx, dy, dz)


def main():
    d = 2**MAGNITUDE
    my, mx = np.mgrid[-np.pi:np.pi:(d * 1j), -np.pi:np.pi:(d * 1j)]
    z = np.cos(my * 4) + np.sin(mx * 4)
    z *= d
    tree = SimpleQuadTree(z)
    fig, axes = plt.subplots(2, 1)

    for azimuth in np.linspace(0, 2 * np.pi, 13):
    # for azimuth in [4/3 * np.pi]:
        plt.cla()
        s = np.ones(z.shape, dtype=bool)
        raycast(tree, z, s, azimuth=azimuth)
        axes[0].set_title(azimuth)
        axes[0].imshow(z, cmap="terrain")
        axes[1].imshow(s)
        # plt.show()
        plt.pause(1.0)


if __name__ == "__main__":
    main()
