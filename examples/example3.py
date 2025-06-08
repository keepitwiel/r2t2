import taichi as ti
import numpy as np

from r2t2 import MinimalRenderer
from example1 import example_map_1


def run(renderer: MinimalRenderer):
    window_shape = (renderer.w_map, renderer.h_map)
    window = ti.ui.Window(name='Example 3', res=window_shape, fps_limit=300, pos=(0, 0))
    illumination_field = np.zeros(shape=window_shape, dtype=np.float32)
    gui = window.get_gui()
    canvas = window.get_canvas()

    azimuth = 0.0
    altitude = np.pi / 4.0
    azimuth_speed = 0.01 # rad per render

    while window.running:
        renderer.render(illumination_field, azimuth, altitude)
        out_image = (illumination_field * 255).astype(np.uint8)
        canvas.set_image(out_image)
        window.show()

        # show GUIs
        with gui.sub_window("Emitter", 0.5, 0.0, width=0.4, height=0.25):
            azimuth = gui.slider_float(f"Azimuth (degrees)", azimuth, 0.0, 2.0 * np.pi)
            azimuth_speed = gui.slider_float(f"Azimuth rotation speed (degrees per frame)", azimuth_speed, -0.1, 0.1)
            altitude = gui.slider_float("Altitude (degrees)", altitude, 0.0, np.pi / 2.0)

        # note that the speed is not adjusted by frame rate
        azimuth = (azimuth + azimuth_speed) % (2.0 * np.pi)


def main(n):
    z, c = example_map_1(n)
    renderer = MinimalRenderer(height_map=z)
    run(renderer)


if __name__ == "__main__":
    ti.init(arch=ti.vulkan)
    main(n=1024)
