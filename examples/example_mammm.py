import numpy as np

from r2t2.maximum_altitude_mmm_np import max_tangent, get_mipmap


def render(output_field, height_field):
    w, h = output_field.shape
    maxmipmap = get_mipmap(height_field, is_max=True)
    dx, dy = -0.5, np.sqrt(3/4)
    tangent = 0.5
    for i in range(w):
        for j in range(h):
            x = i + 0.5
            y = j + 0.5
            output_field[i, j] = max_tangent(x, y, dx, dy, tangent, height_field, maxmipmap)


def main():
    n_cells = 16
    n_nodes = n_cells + 1
    n_complex = n_nodes * 1j
    mx, my = np.mgrid[0:1:n_complex, 0:1:n_complex]
    z = np.cos(mx * np.pi * 4) + np.sin(my * np.pi * 4) + mx ** 2 + my ** 2

    tan_field = np.zeros((n_cells, n_cells))
    render(tan_field, z)

    pass


if __name__ == "__main__":
    main()
