import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

from r2t2.m4.m4 import M4
from r2t2.m4.algorithm import algorithm
from height import get_simple_field


def main():
    n_cells = 4
    z = np.array(
        [
            [ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ]
    ).astype(np.float32)

    m4 = M4(n_cells, np.array([]), np.array([]), z)

    print(f"FLAT MAXMIPMAP: {m4.flat_maxmipmap}")
    print(f"ORIGINAL MAXMIPMAP: {m4.maxmipmap}")
    print("\n\n\n")
    n_levels = m4.get_n_levels()
    offset = 0
    dimension = 2 ** n_levels
    for level in range(n_levels):
        next_offset = offset + dimension**2

        print(f"LEVEL {level}, DIMENSION {dimension}, OFFSET {offset}, NEXT_OFFSET {next_offset}")
        extracted = m4.flat_maxmipmap[offset:next_offset]
        original = m4.maxmipmap[level]
        print(f"extracted flat maxmipmap: {extracted}")
        print(f"original maxmipmap: {original}")
        assert np.all(extracted == original.ravel())
        for i in range(dimension):
            for j in range(dimension):
                assert extracted[i * dimension + j] == original[i, j]

        offset = next_offset
        dimension = dimension // 2
        print("\n")


if __name__ == "__main__":
    ti.init(ti.cpu)
    main()
