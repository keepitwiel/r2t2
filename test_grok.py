import numpy as np
import cv2
from scipy.datasets import face
from max_mip_map import MaxMipMap


if __name__ == "__main__":
    image = face(gray=True)

    maxmipmap = MaxMipMap(image)

    for k in range(maxmipmap.levels):
        result = maxmipmap.get_value(0, 0, k)
        cv2.imshow(f"level {k}: {result.shape[0]} x {result.shape[1]}", result)

    cv2.waitKey()