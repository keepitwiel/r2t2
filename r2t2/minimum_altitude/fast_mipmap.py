from time import time

import numpy as np
import taichi as ti


@ti.func
def reduce(a: float, b: float, c: float, d: float, is_max: bool):
    return max(max(max(a, b), c), d) if is_max else min(min(min(a, b), c), d)


@ti.kernel
def fast_mipmap(inp: ti.types.ndarray(), out_min: ti.types.ndarray(), out_max: ti.types.ndarray()):
    """
    Creates a maxmipmap from an input array
    """
    w, h = inp.shape
    n2, n = out_min.shape
    assert w == h, "input width and height have to be the same"
    assert n * 2 == n2, "output array has to have width = double height"
    assert n == h - 1, "output array height has to be same as input height minus one"
    assert out_min.shape == out_max.shape, "minmipmap and maxmipmap arrays have to have same shape"

    # first loop
    for i in range(n):
        for j in range(n):
            out_max[i, j] = reduce(inp[i, j], inp[i + 1, j], inp[i, j + 1], inp[i + 1, j + 1], is_max=True)
            out_min[i, j] = reduce(inp[i, j], inp[i + 1, j], inp[i, j + 1], inp[i + 1, j + 1], is_max=False)

    # second loop
    step = 2
    source_offset = 0
    target_offset = n
    while step <= n:
        m = n // step
        for i in range(m):
            for j in range(m):
                out_max[i + target_offset, j] = reduce(
                    out_max[i * 2 + source_offset, j * 2],
                    out_max[i * 2 + source_offset + 1, j * 2],
                    out_max[i * 2 + source_offset, j * 2 + 1],
                    out_max[i * 2 + source_offset + 1, j * 2 + 1],
                    is_max=True,
                )
                out_min[i + target_offset, j] = reduce(
                    out_max[i * 2 + source_offset, j * 2],
                    out_max[i * 2 + source_offset + 1, j * 2],
                    out_max[i * 2 + source_offset, j * 2 + 1],
                    out_max[i * 2 + source_offset + 1, j * 2 + 1],
                    is_max=False,
                )
        source_offset = target_offset
        target_offset += m
        step *= 2


def main(n_cells):
    for i in range(100):
        height_field = np.random.uniform(size=(n_cells + 1, n_cells + 1)).astype(np.float32)
        maxmipmap = -np.inf * np.zeros((n_cells * 2 - 1, n_cells), dtype=np.float32)
        minmipmap = np.inf * np.zeros((n_cells * 2 - 1, n_cells), dtype=np.float32)
        t0 = time()
        fast_mipmap(height_field, minmipmap, maxmipmap)
        t1 = time()
        print(f"time: {(t1 - t0):.8f} s")


if __name__ == "__main__":
    ti.init(ti.cpu)
    main(n_cells=128)
