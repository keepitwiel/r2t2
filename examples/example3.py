from dataclasses import field

import taichi as ti
import numpy as np

from r2t2.m4.m4 import M4
from r2t2.m4.algorithm import algorithm
from height import get_simple_field


def main(n_cells: int) -> None:
    window = ti.ui.Window("example3", res=(n_cells, n_cells))
    canvas = window.get_canvas()

    field_size = 1.0
    azimuth = 0
    altitude = 0
    rad_per_frame = np.pi / 360

    mx, my, z = get_simple_field(field_size=field_size, n_cells=n_cells, amplitude=1.0)
    m4 = M4(n_cells, mx, my, z)

    while window.running:
        result = algorithm(m4=m4, azimuth=azimuth, altitude=altitude, cell_size=field_size / n_cells)
        canvas.set_image((result[-1] * 255).astype(np.uint8))
        window.show()
        azimuth += rad_per_frame
        azimuth = azimuth % (2 * np.pi)


if __name__ == "__main__":
    ti.init(ti.cpu)
    main(n_cells=2**10)
