from scipy.datasets import face
from tqdm import tqdm


from simple_relief_mapper import SimpleReliefMapper


def main():
    height_map = face(gray=True)
    renderer = SimpleReliefMapper(height_map)
    pbar = tqdm()
    while True:
        renderer.render(dx=1.0, dy=1.0, dz=1.0, classic=False, scale=1.0)
        pbar.update()


if __name__ == "__main__":
    main()
