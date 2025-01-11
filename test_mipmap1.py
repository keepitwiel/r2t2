import time

import numpy as np
from scipy.datasets import face
from tqdm import tqdm
import cv2

from simple_relief_mapper import SimpleReliefMapper


def main():
    height_map = face(gray=True)[::2, ::2]
    renderer = SimpleReliefMapper(height_map)
    mipmap = renderer.maxmipmap.to_numpy().astype(np.uint8)

    print("simple test")
    print(renderer.get_mipmap_value_python_scope(0, 0, 9), np.max(height_map))
    print(renderer.get_mipmap_value_python_scope(0, 0, 8), np.max(height_map[0:256, 0:256]))
    print(renderer.get_mipmap_value_python_scope(256, 256, 8), np.max(height_map[256:512, 256:512]))

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

    print("calling method, level 0")
    print(renderer.get_mipmap_value_python_scope(0, 0, 0), height_map[0, 0])
    print(renderer.get_mipmap_value_python_scope(1, 0, 0), height_map[1, 0])
    print(renderer.get_mipmap_value_python_scope(0, 1, 0), height_map[0, 1])
    print(renderer.get_mipmap_value_python_scope(383, 511, 0), height_map[383, 511])

    print("calling method, level 1")
    print(renderer.get_mipmap_value_python_scope(0, 0, 1), np.max(height_map[0:2, 0:2]))
    print(renderer.get_mipmap_value_python_scope(2, 2, 1), np.max(height_map[2:4, 2:4]))
    print(renderer.get_mipmap_value_python_scope(3, 3, 1), np.max(height_map[2:4, 2:4]))

    print("calling method, level 2")
    print(renderer.get_mipmap_value_python_scope(0, 0, 2), np.max(height_map[0:4, 0:4]))
    print(renderer.get_mipmap_value_python_scope(4, 4, 2), np.max(height_map[4:8, 4:8]))
    print(renderer.get_mipmap_value_python_scope(8, 8, 2), np.max(height_map[8:12, 8:12]))
    print(renderer.get_mipmap_value_python_scope(9, 9, 2), np.max(height_map[8:12, 8:12]))

    print("calling method, level 3")
    print(renderer.get_mipmap_value_python_scope(0, 0, 3), np.max(height_map[0:8, 0:8]))
    print(renderer.get_mipmap_value_python_scope(8, 8, 3), np.max(height_map[8:16, 8:16]))
    print(renderer.get_mipmap_value_python_scope(9, 9, 3), np.max(height_map[8:16, 8:16]))
    #
    # print(renderer.get_mipmap_value_python_scope(767, 1023, level=0), height_map[767, 1023])
    # print(renderer.get_mipmap_value_python_scope(767, 1023, level=1), np.max(height_map[766:768, 1022:1024]))
    # print(renderer.get_mipmap_value_python_scope(767, 1023, level=2), np.max(height_map[764:768, 1020:1024]))

    cv2.imshow("mipmap", mipmap)
    cv2.waitKey()


if __name__ == "__main__":
    main()
