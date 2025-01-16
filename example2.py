import taichi as ti
from scipy.datasets import face
from tqdm import tqdm

from simple_relief_mapper import SimpleReliefMapper


def main():
    """
    A simple example that shows the framerate, nothing else.
    """
    height_map = face(gray=True)
    renderer = SimpleReliefMapper(height_map)
    pbar = tqdm()
    while True:
        renderer.render(
            azimuth=45,
            altitude=45,
            maxmipmap=False,
            zoom=1.0,
            spp=1,
            sun_width=0.0,
            sun_color=ti.Vector([1.0, 0.9, 0.0]),
            sky_color=ti.Vector([0.2, 0.2, 1.0]),
        )
        pbar.update()


if __name__ == "__main__":
    ti.init(ti.cpu)
    main()
