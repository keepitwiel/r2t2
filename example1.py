import taichi as ti
import numpy as np

from simple_relief_mapper import SimpleReliefMapper
from simple_relief_mapper import simplex_height_map


def run(renderer: SimpleReliefMapper):
    window = ti.ui.Window(name='Example 1', res=(1200, 600), fps_limit=60, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()
    out_image = np.zeros((1200, 600, 3), dtype=np.float32)

    # camera
    zoom = 1.0
    show_maxmipmap = False
    spp = 1

    # algorithm
    l_max_max = 2**renderer.n_levels
    l_max = l_max_max
    maxmipmap = False
    random_xy = True

    # sun
    azimuth = 300.0 # light source horizontal direction, degrees
    azimuth_speed = 0.0 # degrees per render
    altitude = 45.0 # light source vertical direction, degrees
    sun_width = 0.0
    sun_color = (1.0, 0.9, 0.0)

    # sky
    sky_color = (0.2, 0.2, 1.0)

    auto_render = True
    while window.running:
        # x, y = window.get_cursor_pos()
        # if x < 512 / 1200 and y < 512 / 600:
        #     auto_render = False
        #     with gui.sub_window("debug", x, 1.0 - y, 0.2, 0.5):
        #         i = int(x * 1200)
        #         j = int(y * 600)
        #         gui.text(f"{x:0.2f}, {y:0.2f}, {i}, {j}")
        #         n_levels = renderer.n_levels if maxmipmap else 1
        #         counter, l, h, step = renderer.maxmipmap_debug(
        #             i, j, zoom, azimuth, altitude, n_levels
        #         )
        #         gui.text(f"Height at {i}, {j}: {h:0.2f}")
        #         gui.text(f"maxmipmap level 1: {renderer.maxmipmap[i // 2, 0 + j // 2]}")
        #         gui.text(f"maxmipmap level 2: {renderer.maxmipmap[i // 4, 256 + j // 4]}")
        #         gui.text(f"maxmipmap level 3: {renderer.maxmipmap[i // 8, 384 + j // 8]}")
        #         gui.text(f"maxmipmap level 4: {renderer.maxmipmap[i // 16, 448 + j // 16]}")
        #         gui.text(f"maxmipmap level 5: {renderer.maxmipmap[i // 32, 480 + j // 32]}")
        #         gui.text(f"maxmipmap level 6: {renderer.maxmipmap[i // 64, 496 + j // 64]}")
        #         gui.text(f"Number of steps: {counter}")
        #         gui.text(f"Ray length: {l:0.2f}")
        #         gui.text(f"Last step size: {step}")
        # else:
        #     auto_render = True

        with gui.sub_window("Camera", 0.5, 0.1, 0.5, 0.2):
            zoom = gui.slider_float("Zoom", zoom, 0.1, 10.0)
            show_maxmipmap = gui.checkbox("Show MaxMipMap", show_maxmipmap)
            spp = gui.slider_int("Samples per pixel", spp, 1, 16)

        with gui.sub_window("Algorithm", 0.5, 0.3, 0.5, 0.2):
            maxmipmap = gui.checkbox("Enable MaxMipMap", maxmipmap)
            l_max = gui.slider_float("Maximum ray length", l_max, 0.0, l_max_max)
            random_xy = gui.checkbox("Randomize ray spawn point within pixel", random_xy)

        with gui.sub_window("Sun", 0.5, 0.5, 0.5, 0.2):
            azimuth = gui.slider_float(f"Azimuth (degrees)", azimuth, 0, 360)
            azimuth_speed = gui.slider_float(f"Azimuth rotation speed (degrees)", azimuth_speed, -5, 5)
            altitude = gui.slider_float("Altitude (degrees)", altitude, 0, 90)
            sun_width = gui.slider_float("Sun width (degrees)", sun_width, 0.0, 5.0)
            sun_color = gui.color_edit_3("Color", sun_color)

        with gui.sub_window("Sky", 0.5, 0.7, 0.5, 0.2):
            sky_color = gui.color_edit_3("Color", sky_color)

        if auto_render:
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
            # the following rotates the azimuth between 0 and 360 degrees, with increments of 1 degree per step
            azimuth = (azimuth + azimuth_speed) % 360

        if show_maxmipmap:
            out_image[:renderer.w // 2, :renderer.w - 1, 0] = renderer.maxmipmap.to_numpy().astype(np.float32)
            out_image[:renderer.w // 2, :renderer.w - 1, 1] = renderer.maxmipmap.to_numpy().astype(np.float32)
            out_image[:renderer.w // 2, :renderer.w - 1, 2] = renderer.maxmipmap.to_numpy().astype(np.float32)

            # out_image[:renderer.w, :renderer.h, 0] = renderer.height_map.to_numpy().astype(np.float32)
            # out_image[:renderer.w, :renderer.h, 1] = renderer.height_map.to_numpy().astype(np.float32)
            # out_image[:renderer.w, :renderer.h, 2] = renderer.height_map.to_numpy().astype(np.float32)
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
    # define height map
    z, c = example_map_1(n)

    # initialize renderer
    renderer = SimpleReliefMapper(height_map=z, map_color=c, cell_size=1.0)

    # run app
    run(renderer)


if __name__ == "__main__":
    ti.init(arch=ti.vulkan)
    main(n=512)
