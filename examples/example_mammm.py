import numpy as np
import matplotlib.pyplot as plt

from r2t2.maximize_tangent.maximize_tangent_np import max_tangent, get_mipmap


def render(output_field, height_field):
    w, h = output_field.shape
    maxmipmap = get_mipmap(height_field, is_max=True)
    dx, dy = -0.5, np.sqrt(3/4)
    alt = np.pi / 6
    radius = np.pi / 12
    tan0 = np.tan(alt)
    tan1 = np.tan(alt + radius)
    for i in range(w):
        for j in range(h):
            x = i + 0.5
            y = j + 0.5
            output_field[i, j] = max_tangent(x, y, dx, dy, tan0, tan1, height_field, maxmipmap)


def main():
    n_cells = 64
    n_nodes = n_cells + 1
    n_complex = n_nodes * 1j
    mx, my = np.mgrid[0:1:n_complex, 0:1:n_complex]
    z = n_cells / 16 * (np.cos(mx * np.pi * 8) + np.sin(my * np.pi * 8) + mx ** 2 + my ** 2)

    tan_field = np.zeros((n_cells, n_cells))
    render(tan_field, z)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(z, interpolation="bilinear", cmap="terrain")
    axes[1].imshow(tan_field, origin="lower")
    plt.show()


if __name__ == "__main__":
    main()
