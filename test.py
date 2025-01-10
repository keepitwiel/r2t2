import numpy as np
import cv2
from scipy.datasets import face


def get_maxmipmap(z):
    w, h = z.shape
    result = np.zeros(shape=(w // 2, h), dtype=float)
    y0 = 0
    while min(z.shape) > 1:
        w, h = z.shape
        w_, h_ = w // 2, h // 2
        mipmap = np.zeros(shape=(w_, h_), dtype=float)
        for i in range(0, w, 2):
            for j in range(0, h, 2):
                mipmap[i // 2][j // 2] = np.max(z[i:i + 2, j:j + 2])
        result[0:w_, y0:y0 + h_] = mipmap
        z = mipmap
        y0 += h_

    # final level: single value
    mipmap[0, y0] = np.max(z)
    return result


if __name__ == "__main__":
    image = face(gray=True)
    mmm = get_maxmipmap(image)
    cv2.imshow("foo", mmm)
