import taichi as ti
import numpy as np

from r2t2 import Renderer
from height import simplex_height_map


def run(renderer: Renderer):
    window = ti.ui.Window(name='Example 1', res=(1200, 600), fps_limit=60, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()
    out_image = np.zeros((1200, 600, 3), dtype=np.float32)

    # camera
    zoom = 1.0
    x_offset = 0.0
    y_offset = 0.0
    show_maxmipmap = False
    spp = 4

    # algorithm
    l_max_max = 2**renderer.n_levels
    l_max = l_max_max
    random_xy = True

    # sun
    azimuth = 300.0 # light source horizontal direction, degrees
    azimuth_speed = 0.0 # degrees per render
    altitude = 45.0 # light source vertical direction, degrees
    sun_radius = 10.0
    sun_color = (1.0, 0.9, 0.0)

    # sky
    sky_color = (0.2, 0.2, 1.0)

    while window.running:
        with gui.sub_window("Camera", 0.5, 0.1, width=0.4, height=0.2):
            zoom = gui.slider_float("Zoom", zoom, 0.1, 10.0)
            x_offset = gui.slider_float("X offset", x_offset, -renderer.w, renderer.w)
            y_offset = gui.slider_float("Y offset", y_offset, -renderer.h, renderer.h)
            show_maxmipmap = gui.checkbox("Show MaxMipMap", show_maxmipmap)
            spp = gui.slider_int("Samples per pixel", spp, 1, 16)

        with gui.sub_window("Algorithm", 0.5, 0.3, width=0.4, height=0.15):
            l_max = gui.slider_float("Maximum ray length", l_max, 0.0, l_max_max)
            random_xy = gui.checkbox("Randomize ray spawn point within pixel", random_xy)

        with gui.sub_window("Sun", 0.5, 0.5, width=0.4, height=0.25):
            azimuth = gui.slider_float(f"Azimuth (degrees)", azimuth, 0, 360)
            azimuth_speed = gui.slider_float(f"Azimuth rotation speed (degrees)", azimuth_speed, -5, 5)
            altitude = gui.slider_float("Altitude (degrees)", altitude, 0, 90)
            sun_radius = gui.slider_float("Sun radius (degrees)", sun_radius, 0.0, 5.0)
            sun_color = gui.color_edit_3("Color", sun_color)

        with gui.sub_window("Sky", 0.5, 0.8, width=0.4, height=0.1):
            sky_color = gui.color_edit_3("Color", sky_color)

        renderer.render(
            azimuth,
            altitude,
            zoom,
            x_offset,
            y_offset,
            spp=spp,
            sun_radius=sun_radius,
            sun_color=sun_color,
            sky_color=sky_color,
            l_max=l_max,
            random_xy=random_xy,
        )
        # the following rotates the azimuth between 0 and 360 degrees, with increments of 1 degree per step
        # note that the speed is not adjusted by frame rate
        azimuth = (azimuth + azimuth_speed) % 360

        if show_maxmipmap:
            mmm = renderer.maxmipmap.to_numpy().astype(np.float32)
            out_image[:renderer.w // 2, :renderer.w - 1, 0] = mmm
            out_image[:renderer.w // 2, :renderer.w - 1, 1] = mmm
            out_image[:renderer.w // 2, :renderer.w - 1, 2] = mmm
            canvas.set_image((out_image - np.float32(renderer.min_value)) / np.float32(renderer.max_value - renderer.min_value))
        else:
            out_image[:renderer.w, :renderer.h] = renderer.get_image()
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
    renderer = Renderer(height_map=z, map_color=c)
    run(renderer)


if __name__ == "__main__":
    ti.init(arch=ti.vulkan)
    main(n=512)
