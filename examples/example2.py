import taichi as ti
from scipy.datasets import face
from tqdm import tqdm

from r2t2 import Renderer


def main():
    """
    A simple example that shows the framerate, nothing else.
    """
    height_map = face(gray=True)
    renderer = Renderer(height_map)
    pbar = tqdm()
    while True:
        renderer.render(
            azimuth=45,
            altitude=45,
            zoom=1.0,
            x_offset=0.0,
            y_offset=0.0,
            spp=1,
            sun_radius=0.0,
            sun_color=ti.Vector([1.0, 0.9, 0.0]),
            sky_color=ti.Vector([0.2, 0.2, 1.0]),
            l_max=max(height_map.shape),
            random_xy=True,
        )
        pbar.update()


if __name__ == "__main__":
    ti.init(ti.vulkan)
    main()
