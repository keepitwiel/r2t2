import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

from r2t2.m4.m4 import M4
from r2t2.m4.algorithm import algorithm
from height import get_simple_field


def plot_shadow(z, shadow):
    fig, axes = plt.subplots(1, 1 + len(shadow))
    axes[0].imshow(z, cmap="terrain")
    for i, s in enumerate(shadow):
        axes[1 + i].imshow(s)
    plt.show()


def main(field_size, n_cells, amplitude, azimuth, altitude):
    mx, my, z = get_simple_field(field_size, n_cells, amplitude)
    m4 = M4(n_cells, mx, my, z)
    result = algorithm(m4=m4, azimuth=azimuth, altitude=altitude, cell_size=field_size / n_cells)
    plot_shadow(z, result)


if __name__ == "__main__":
    ti.init(ti.cpu)
    main(field_size=1.0, n_cells=2**7, amplitude=1.0, azimuth=np.pi / 4, altitude=np.pi / 4)
