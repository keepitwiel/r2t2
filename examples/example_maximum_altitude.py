import numpy as np
import taichi as ti

from r2t2 import render_maximum_altitude


def main(n_cells: int):
    window_shape = (800, 800)
    window = ti.ui.Window(name='Maximum Altitude example', res=window_shape, fps_limit=300, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()

    theta = 0.0
    n_nodes = (n_cells + 1) * 1j
    mx, my = np.mgrid[:1:n_nodes, :1:n_nodes]
    azi = np.pi / 4
    alt = np.pi / 12
    radius = np.pi / 180 # 1 degree
    log_min = -2
    log_max = int(np.log2(n_cells))
    illumination_field = np.zeros((n_cells, n_cells), dtype=np.float32)
    while window.running:
        height_field = n_cells / 4 * (np.sin(mx * 4 * np.pi + theta) + np.cos(my * 4 * np.pi + theta)).astype(
            np.float32)
        render_maximum_altitude(illumination_field, height_field, log_min, log_max, azi, alt, radius)
        canvas.set_image(illumination_field.astype(np.float32))
        window.show()

        with gui.sub_window("angles", 0.5, 0.0, 0.5, 0.3):
            alt = gui.slider_float("Minimum altitude (rad)", alt, 0.0, np.pi / 2 - radius)
            radius = gui.slider_float("Emitter radius (rad)", radius, 0.0, np.pi / 4)
            azi = gui.slider_float("Azimuth (rad)", azi, 0.0, 2 * np.pi)
            log_min = gui.slider_int("log_min", log_min, -4, 2)
            log_max = gui.slider_int("log_max", log_max, 4, int(np.log2(n_cells)))
        theta += np.pi / 180
        theta = theta % (2.0 * np.pi)


if __name__ == "__main__":
    ti.init(ti.vulkan)
    main(n_cells=1024)
