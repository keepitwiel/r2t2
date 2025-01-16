import taichi as ti
import numpy as np
from scipy.datasets import face
import cv2

from simple_relief_mapper import SimpleReliefMapper


def main():
    height_map = face(gray=True)[::2, ::2]
    renderer = SimpleReliefMapper(height_map)
    mipmap = renderer.maxmipmap.to_numpy().astype(np.uint8)

    print("level 0")
    print(mipmap[0, 0], np.max(height_map[0:2, 0:2]))
    print(mipmap[1, 0], np.max(height_map[2:4, 0:2]))
    print(mipmap[0, 1], np.max(height_map[0:2, 2:4]))
    print(mipmap[1, 1], np.max(height_map[2:4, 2:4]))
    print(mipmap[2, 0], np.max(height_map[4:6, 0:2]))
    print(mipmap[2, 1], np.max(height_map[4:6, 2:4]))
    print(mipmap[1, 2], np.max(height_map[2:4, 4:6]))

    print("level 1")
    print(mipmap[0, 256 + 0], np.max(height_map[0:4, 0:4]))
    print(mipmap[0, 256 + 1], np.max(height_map[0:4, 4:8]))
    print(mipmap[1, 256 + 0], np.max(height_map[4:8, 0:4]))
    print(mipmap[1, 256 + 1], np.max(height_map[4:8, 4:8]))
    print(mipmap[0, 256 + 2], np.max(height_map[0:4, 8:12]))
    print(mipmap[1, 256 + 2], np.max(height_map[4:8, 8:12]))
    print(mipmap[2, 256 + 1], np.max(height_map[8:12, 4:8]))

    cv2.imshow("mipmap", mipmap)
    cv2.waitKey()


if __name__ == "__main__":
    ti.init(ti.cpu)
    main()
