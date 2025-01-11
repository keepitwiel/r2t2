import taichi as ti
import numpy as np

from simple_relief_mapper import SimpleReliefMapper
from height import simplex_height_map


def run(renderer: SimpleReliefMapper):
    window = ti.ui.Window(name='Window Title', res=renderer.get_shape(), fps_limit=30, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()
    azimuth = 45 # light source horizontal direction, degrees
    altitude = 15 # light source vertical direction, degrees
    classic = True
    while window.running:
        with gui.sub_window("Sub Window", 0.1, 0.1, 0.8, 0.2):
            altitude = gui.slider_float("altitude (deg)", altitude, 0, 89)
            classic = gui.checkbox("classic mode", classic)
        dx, dy, dz = renderer.get_direction(azimuth, altitude)
        renderer.render(dx, dy, dz, classic)
        canvas.set_image(renderer.get_image())
        window.show()

        # the following rotates the azimuth between 0 and 360 degrees, with increments of 1 degree per step
        azimuth = (azimuth + 1) % 360


def example_map_1(n):
    """A height map generated from simplex noise. In the middle, a small square area gets a value of 0."""
    octaves = int(np.log2(n))
    z = simplex_height_map(dim=n, octaves=octaves, amplitude=n, seed=42)
    z = np.float32(z)
    z[n // 2 - n // 8:n // 2 + n // 8, n // 2 - n // 8:n // 2 + n // 8] = 0
    return z


def main(n):
    # define height map
    z = example_map_1(n)

    # initialize renderer
    renderer = SimpleReliefMapper(height_map=z)

    # run app
    run(renderer)


if __name__ == "__main__":
    main(n=512)
