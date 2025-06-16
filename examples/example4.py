import numpy as np
import taichi as ti


from r2t2.minimum_altitude.rendering_pipeline import ShaderPipeline



def main(n_cells: int):
    window_shape = (n_cells, n_cells)
    window = ti.ui.Window(name='Example 3', res=(800, 800), fps_limit=300, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()

    pipeline = ShaderPipeline(n_cells=n_cells, cell_size=1.0)
    theta = 0.0
    n_nodes = (n_cells + 1) * 1j
    mx, my = np.mgrid[:1:n_nodes, :1:n_nodes]
    global_min_altitude = np.pi / 12
    emitter_height = np.pi / 180  # 1 degree
    azimuth = 0 # np.pi / 4
    n_samples = 1

    while window.running:
        height_field = (np.sin(mx * 4 * np.pi + theta) + np.cos(my * 4 * np.pi + theta)).astype(np.float32)
        illumination_field = np.zeros((n_cells, n_cells), dtype=np.float32)
        pipeline.render(
            height_field=height_field,
            azimuth=azimuth,
            global_min_altitude=global_min_altitude,
            global_max_altitude=global_min_altitude + emitter_height,
            n_samples=n_samples,
            illumination_field=illumination_field,
            simple=True,
        )
        canvas.set_image((global_min_altitude + emitter_height - illumination_field) / emitter_height)
        window.show()

        with gui.sub_window("angles", 0.5, 0.0, 0.5, 0.3):
            global_min_altitude = gui.slider_float("Minimum altitude", global_min_altitude, 0.0, np.pi / 2 - emitter_height)
            emitter_height = gui.slider_float("Emitter height", emitter_height, 0.0, np.pi / 12)
            azimuth = gui.slider_float("Azimuth", azimuth, 0.0, 2 * np.pi)
            n_samples = gui.slider_int("n_samples", n_samples, 1, 16)
        theta += np.pi / 180
        theta = theta % (2.0 * np.pi)



if __name__ == "__main__":
    ti.init(ti.cpu)
    main(n_cells=256)
