import taichi as ti
import numpy as np

from r2t2 import Renderer
from height import simplex_height_map


def run(renderer: Renderer):
    window = ti.ui.Window(name='Example 1', res=(1200, 600), fps_limit=300, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()
    out_image = np.zeros((1200, 600, 3), dtype=np.float32)

    # camera
    show_maxmipmap = False
    spp = 4

    # algorithm
    l_max_max = renderer.l_max

    # sun
    azimuth_speed = 0.5 # degrees per render

    while window.running:
        # render the image
        renderer.render()

        # show GUIs
        with gui.sub_window("Camera", 0.5, 0.1, width=0.4, height=0.4):
            renderer.zoom = gui.slider_float("Zoom", renderer.zoom, 0.1, 10.0)
            renderer.x_offset = gui.slider_float("X offset", renderer.x_offset, -renderer.w_map, renderer.w_map)
            renderer.y_offset = gui.slider_float("Y offset", renderer.y_offset, -renderer.h_map, renderer.h_map)
            show_maxmipmap = gui.checkbox("Show MaxMipMap", show_maxmipmap)
            renderer.spp = gui.slider_int("Samples per pixel", renderer.spp, 1, 16)
            renderer.brightness = gui.slider_float("Brightness", renderer.brightness, 0.1, 10.0)

        with gui.sub_window("Algorithm", 0.5, 0.4, width=0.4, height=0.15):
            renderer.l_max = gui.slider_float("Maximum ray length", renderer.l_max, 0.0, l_max_max)
            renderer.random_xy = gui.checkbox("Randomize ray spawn point within pixel", renderer.random_xy)

        with gui.sub_window("Sun", 0.5, 0.6, width=0.4, height=0.25):
            renderer.azimuth = gui.slider_float(f"Azimuth (degrees)", renderer.azimuth, 0, 360)
            azimuth_speed = gui.slider_float(f"Azimuth rotation speed (degrees per frame)", azimuth_speed, -5, 5)
            renderer.altitude = gui.slider_float("Altitude (degrees)", renderer.altitude, 0, 90)
            renderer.sun_radius = gui.slider_float("Sun radius (degrees)", renderer.sun_radius, 0.0, 5.0)
            renderer.sun_color = gui.color_edit_3("Color", renderer.sun_color)

        with gui.sub_window("Sky", 0.5, 0.9, width=0.4, height=0.1):
            renderer.sky_color = gui.color_edit_3("Color", renderer.sky_color)

        # the following rotates the azimuth between 0 and 360 degrees, with increments of 1 degree per step
        # note that the speed is not adjusted by frame rate
        renderer.azimuth = (renderer.azimuth + azimuth_speed) % 360

        # display image
        if show_maxmipmap:
            mmm = renderer.maxmipmap.to_numpy().astype(np.float32)
            out_image[:renderer.w_map // 2, :renderer.w_map - 1, 0] = mmm
            out_image[:renderer.w_map // 2, :renderer.w_map - 1, 1] = mmm
            out_image[:renderer.w_map // 2, :renderer.w_map - 1, 2] = mmm
            canvas.set_image((out_image - np.float32(renderer.min_value)) / np.float32(renderer.max_value - renderer.min_value))
        else:
            out_image[:renderer.w_canvas, :renderer.h_canvas] = renderer.get_image()
            canvas.set_image(out_image)
        window.show()


def example_map_1(n):
    """A height map generated from simplex noise with dimension n.
    In the middle, a small plateau is defined with
    height 0, and inside that plateau a tower
    of height n is placed.
    :param n: map dimension
    :return: a 2D height map in the form of a numpy array.
    """
    octaves = int(np.log2(n))
    z = simplex_height_map(dim=n, octaves=octaves, amplitude=n, seed=42)
    z = np.float32(z)

    # middle plateau
    lb = n // 2 - n // 8
    ub = n // 2 + n // 8
    z[lb:ub, lb:ub] = 0

    # middle tower
    lb = n // 2 - n // 32
    ub = n // 2 + n // 32
    z[lb:ub, lb:ub] = n

    # color
    c = np.random.uniform(0.2, 0.8, size=(n, n)).astype(np.float32)
    c = np.stack([c, c, c], axis=2)
    return z, c


def main(n):
    z, c = example_map_1(n)
    renderer = Renderer(height_map=z, map_color=c, rgb_type="uint8")
    run(renderer)


if __name__ == "__main__":
    ti.init(arch=ti.vulkan)
    main(n=1024)
