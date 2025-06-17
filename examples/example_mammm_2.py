import numpy as np
import taichi as ti

from r2t2.maximize_tangent.maximize_tangent_np import max_tangent, get_mipmap



def render(output_field, height_field, azi, alt, radius):
    w, h = output_field.shape
    maxmipmap = get_mipmap(height_field, is_max=True)
    dx, dy = np.cos(azi), np.sin(azi)
    tan0 = np.tan(alt)
    tan1 = np.tan(alt + radius)
    for i in range(w):
        for j in range(h):
            x = i + 0.5
            y = j + 0.5
            tangent = max_tangent(x, y, dx, dy, tan0, tan1, height_field, maxmipmap)
            if radius > 0:
                output_field[i, j] = (np.arctan(tan1) - np.arctan(tangent)) / radius
            else:
                output_field[i, j] = 1.0 * (tan1 == tangent)

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
    illumination_field = np.zeros((n_cells, n_cells), dtype=np.float32)
    while window.running:
        height_field = n_cells / 4 * (np.sin(mx * 4 * np.pi + theta) + np.cos(my * 4 * np.pi + theta)).astype(
            np.float32)
        render(illumination_field, height_field, azi, alt, radius)
        canvas.set_image(illumination_field.astype(np.float32))
        window.show()

        with gui.sub_window("angles", 0.5, 0.0, 0.5, 0.3):
            alt = gui.slider_float("Minimum altitude (rad)", alt, 0.0, np.pi / 2 - radius)
            radius = gui.slider_float("Emitter radius (rad)", radius, 0.0, np.pi / 4)
            azi = gui.slider_float("Azimuth (rad)", azi, 0.0, 2 * np.pi)
        theta += np.pi / 180
        theta = theta % (2.0 * np.pi)


if __name__ == "__main__":
    ti.init(ti.cpu)
    main(n_cells=128)
