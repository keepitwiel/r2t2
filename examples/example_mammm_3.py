import numpy as np
import taichi as ti

from r2t2.maximize_tangent.maximize_tangent import one_step_mipmap, render


def update(illumination_field, mx, my, height_field, maxmipmap, n_cells, theta, azi, alt, radius, n_levels):
    height_field[:, :] = n_cells / 4 * (np.sin(mx * 8 * np.pi + theta) + np.cos(my * 8 * np.pi + theta)).astype(
        np.float32)
    one_step_mipmap(height_field, maxmipmap)
    render(illumination_field, height_field, maxmipmap, azi, alt, radius, n_levels)
    theta += np.pi / 180
    theta = theta % (2.0 * np.pi)
    return theta


def main(n_cells: int):
    window_shape = (800, 800)
    window = ti.ui.Window(name='Maximum Altitude example', res=window_shape, fps_limit=300, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()
    n_levels = int(np.log2(n_cells))
    assert n_levels == np.log2(n_cells)
    theta = 0.0
    n_nodes = (n_cells + 1)
    azi = np.pi / 4
    alt = np.pi / 12
    radius = np.pi / 180 # 1 degree
    n_grid = n_nodes * 1j
    mx, my = np.mgrid[:1:n_grid, :1:n_grid]
    height_field = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    illumination_field = np.zeros((800, 800), dtype=np.float32)
    maxmipmap = np.zeros((n_cells * 2, n_cells - 1), dtype=np.float32)
    while window.running:
        with gui.sub_window("angles", 0.5, 0.0, 0.5, 0.3):
            alt = gui.slider_float("Minimum altitude (rad)", alt, 0.0, np.pi / 2 - radius)
            radius = gui.slider_float("Emitter radius (rad)", radius, 0.0, np.pi / 4)
            azi = gui.slider_float("Azimuth (rad)", azi, 0.0, 2 * np.pi)
        canvas.set_image(illumination_field[:800, :800].astype(np.float32))
        window.show()
        theta = update(illumination_field, mx, my, height_field, maxmipmap, n_cells, theta, azi, alt, radius, n_levels)



if __name__ == "__main__":
    ti.init(ti.cpu)
    main(n_cells=1024)
