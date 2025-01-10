import taichi as ti
import numpy as np
from scipy.datasets import face

from simple_relief_mapper import SimpleReliefMapper
from height import simplex_height_map


def example_map_1(n):
    """A height map generated from simplex noise. In the middle, a small square area gets a value of 0."""
    octaves = int(np.log2(n))
    z = simplex_height_map(dim=n, octaves=octaves, amplitude=n, seed=42)
    z = np.float32(z)
    z[n // 2 - n // 8:n // 2 + n // 8, n // 2 - n // 8:n // 2 + n // 8] = 0
    return z


def main():
    height_map = face(gray=True)
    renderer = SimpleReliefMapper(height_map)
    print(renderer.get_mipmap_value(0, 0, 0), height_map[0, 0])
    print(renderer.get_mipmap_value(0, 0, 1), np.max(height_map[0:2, 0:2]))
    print(renderer.get_mipmap_value(2, 2, 1), np.max(height_map[2:4, 2:4]))
    print(renderer.get_mipmap_value(3, 3, 1), np.max(height_map[2:4, 2:4]))
    print(renderer.get_mipmap_value(0, 0, 2), np.max(height_map[0:4, 0:4]))
    print(renderer.get_mipmap_value(4, 4, 2), np.max(height_map[4:8, 4:8]))
    print(renderer.get_mipmap_value(8, 8, 2), np.max(height_map[8:12, 8:12]))
    print(renderer.get_mipmap_value(9, 9, 2), np.max(height_map[8:12, 8:12]))
    print(renderer.get_mipmap_value(0, 0, 3), np.max(height_map[0:8, 0:8]))
    print(renderer.get_mipmap_value(8, 8, 3), np.max(height_map[8:16, 8:16]))
    print(renderer.get_mipmap_value(9, 9, 3), np.max(height_map[8:16, 8:16]))

    renderer.get_mipmap_value(767, 1023, level=0)
    renderer.get_mipmap_value(767, 1023, level=1)
    renderer.get_mipmap_value(767, 1023, level=10)


if __name__ == "__main__":
    main()
