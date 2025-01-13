import time

import numpy as np
from scipy.datasets import face
from tqdm import tqdm
import cv2

from simple_relief_mapper import SimpleReliefMapper


def main():
    height_map = face(gray=True)[::256, ::256].astype(np.float32)
    renderer = SimpleReliefMapper(height_map)
    mipmap = renderer.maxmipmap.to_numpy()

    print("height map:")
    print(height_map)

    print("Mipmap:")
    print(mipmap)

    print("level 2")
    print(renderer.get_mipmap_value_python_scope(0, 0, 2), np.max(height_map))

    print("level 1")
    print(renderer.get_mipmap_value_python_scope(0, 0, 1), np.max(height_map[0:2, 0:2]))
    print(renderer.get_mipmap_value_python_scope(0, 2, 1), np.max(height_map[0:2, 2:4]))
    print(renderer.get_mipmap_value_python_scope(2, 0, 1), np.max(height_map[2:4, 0:2]))
    print(renderer.get_mipmap_value_python_scope(2, 2, 1), np.max(height_map[2:4, 2:4]))

    print("level 0")
    for i in [0, 1, 2]:
        for j in [0, 1, 2, 3]:
            print(renderer.get_mipmap_value_python_scope(i, j, 0), height_map[i, j])

    cv2.imshow("mipmap", mipmap)
    cv2.waitKey()


if __name__ == "__main__":
    main()
