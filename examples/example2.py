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
        renderer.render(map_color)
        pbar.update()


if __name__ == "__main__":
    ti.init(ti.vulkan)
    main()
