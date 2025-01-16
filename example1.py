import taichi as ti
import numpy as np

from simple_relief_mapper import SimpleReliefMapper
from simple_relief_mapper import simplex_height_map


def run(renderer: SimpleReliefMapper):
    window = ti.ui.Window(name='Example 1', res=(1200, 600), fps_limit=60, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()

    zoom = 1.0
    height_map = False
    spp = 1
    l_max_max = 2**renderer.n_levels
    l_max = l_max_max

    maxmipmap = False
    random_xy = True

    azimuth = 45.0 # light source horizontal direction, degrees
    altitude = 45.0 # light source vertical direction, degrees
    sun_width = 0.0
    sun_color = (1.0, 0.9, 0.0)

    sky_color = (0.2, 0.2, 1.0)

    out_image = np.zeros((1200, 600, 3), dtype=np.float32)

    while window.running:
        with gui.sub_window("Camera", 0.5, 0.1, 0.5, 0.2):
            zoom = gui.slider_float("Zoom", zoom, 0.1, 10.0)
            height_map = gui.checkbox("Show height map", height_map)
            spp = gui.slider_int("Samples per pixel", spp, 1, 16)

        with gui.sub_window("Algorithm", 0.5, 0.3, 0.5, 0.2):
            maxmipmap = gui.checkbox("Enable MaxMipMap", maxmipmap)
            l_max = gui.slider_float("Maximum ray length", l_max, 0.0, l_max_max)
            random_xy = gui.checkbox("Randomize ray spawn point within pixel", random_xy)

        with gui.sub_window("Sun", 0.5, 0.5, 0.5, 0.2):
            gui.text(f"Azimuth: {azimuth:.0f} degrees")
            altitude = gui.slider_float("Altitude (degrees)", altitude, 0, 89)
            sun_width = gui.slider_float("Sun width (degrees)", sun_width, 0.0, 5.0)
            sun_color = gui.color_edit_3("Color", sun_color)

        with gui.sub_window("Sky", 0.5, 0.7, 0.5, 0.2):
            sky_color = gui.color_edit_3("Color", sky_color)

        renderer.render(
            azimuth,
            altitude,
            maxmipmap,
            zoom,
            spp=spp,
            sun_width=sun_width,
            sun_color=sun_color,
            sky_color=sky_color,
            l_max=l_max,
            random_xy=random_xy,
        )
        if height_map:
            out_image[:renderer.w, :renderer.h, 0] = renderer.height_map.to_numpy().astype(np.float32)
            out_image[:renderer.w, :renderer.h, 1] = renderer.height_map.to_numpy().astype(np.float32)
            out_image[:renderer.w, :renderer.h, 2] = renderer.height_map.to_numpy().astype(np.float32)
            canvas.set_image((out_image - np.float32(renderer.min_value)) / np.float32(renderer.max_value - renderer.min_value))
        else:
            out_image[:renderer.w, :renderer.h] = renderer.get_image()
            canvas.set_image(out_image)
        window.show()

        # the following rotates the azimuth between 0 and 360 degrees, with increments of 1 degree per step
        azimuth = (azimuth + 1) % 360


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
    z[n // 2 - n // 8:n // 2 + n // 8, n // 2 - n // 8:n // 2 + n // 8] = 0

    # middle tower
    z[n // 2 - n // 32:n // 2 + n // 32, n // 2 - n // 32:n // 2 + n // 32] = n

    # color
    c = np.random.uniform(0.2, 0.8, size=(n, n)).astype(np.float32)
    c = np.stack([c, c, c], axis=2)
    return z, c


def main(n):
    # define height map
    z, c = example_map_1(n)

    # initialize renderer
    renderer = SimpleReliefMapper(height_map=z, map_color=c, cell_size=1.0)

    # run app
    run(renderer)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    main(n=512)
