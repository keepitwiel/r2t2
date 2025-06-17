import numpy as np
import taichi as ti


@ti.kernel
def fast_mipmap(inp: ti.types.ndarray(), out: ti.types.ndarray()):
    """
    Creates a maxmipmap from an input array
    """
    w, h = inp.shape
    n2, n = out.shape
    assert w == h, "input width and height have to be the same"
    assert n * 2 == n2, "output array has to have width = double height"
    assert n == h - 1, "output array height has to be same as input height minus one"

    # first loop
    for i in range(n):
        for j in range(n):
            out[i, j] = max(inp[i, j], inp[i + 1, j], inp[i, j + 1], inp[i + 1, j + 1])

    # second loop
    step = 2
    source_offset = 0
    target_offset = n
    while step <= n:
        m = n // step
        for i in range(m):
            for j in range(m):
                out[i + target_offset, j] = max(
                    out[i * 2 + source_offset, j * 2],
                    out[i * 2 + source_offset + 1, j * 2],
                    out[i * 2 + source_offset, j * 2 + 1],
                    out[i * 2 + source_offset + 1, j * 2 + 1],
                )
        source_offset = target_offset
        target_offset += m
        step *= 2


#
# def reduce(array: np.ndarray, step_size: int, is_max: bool) -> np.ndarray:
#     if is_max:
#         func = np.max
#     else:
#         func = np.min
#     result = func(
#         np.stack(
#             [
#                 array[:-1:step_size, :-1:step_size],
#                 array[1::step_size, :-1:step_size],
#                 array[:-1:step_size, 1::step_size],
#                 array[1::step_size, 1::step_size],
#             ],
#             axis=2,
#         ),
#         axis=2,
#     )
#     return result
#
#
# def get_mipmap(z: np.ndarray, is_max: bool) -> list[np.ndarray]:
#     w, h = z.shape
#     assert w == h
#     n_levels = int(np.log2(w - 1))
#     buffer = reduce(z, step_size=1, is_max=is_max)
#     result = [buffer]
#     for level in range(n_levels):
#         buffer = reduce(buffer, step_size=2, is_max=is_max)
#         result.append(buffer)
#     return result
#
#
# @ti.func
# def get_max_height(maxmipmap: ti.types.ndarray(), x: float, y: float):


@ti.func
def get_height(height_field: ti.types.ndarray(), x: float, y: float):
    w, h = height_field.shape
    result = -np.inf
    if 0 <= x < w - 1 and 0 <= y < h - 1:
        i, j = int(x), int(y)
        u, v = x % 1, y % 1
        z00 = height_field[i, j]
        z10 = height_field[i + 1, j]
        z01 = height_field[i, j + 1]
        z11 = height_field[i + 1, j + 1]
        result = (1 - u) * (1 - v) * z00 + u * (1 - v) * z10 + (1 - u) * v * z01 + u * v * z11
    return result



@ti.func
def get_max_level(maxmipmap: ti.types.ndarray(), x: float, y: float, z: float, dx: float, dy: float, n_levels: int):
    step_size = 1
    # level = 0
    l_ = 0
    offset = 0
    for l in range(n_levels):
        l_ = l
        d = 2**(n_levels - l)
        i, j = int(x // step_size), int(y // step_size)
        if i == x // step_size:
            if dx < 0:
                i -= 1
        if j == y // step_size:
            if dy < 0:
                j -= 1
        if 0 <= i < d and 0 <= j < d:
            z_max = maxmipmap[offset + i, j]
            if z < z_max:
                l_ -= 1
                break
        else:
            # we're outside the domain - we are above the terrain by definition
            l_ = n_levels
        step_size *= 2
        offset += d
    return l_


@ti.func
def partial_dt(x: float, dx: float, cell_size: int):
    """
    example 1: x = 1, dx = -0.5, cell_size = 1:
        i = 1 - 1 = 0
        t = (0 * cell_size - x) / dx = -1 / -0.5 = 2

    example 2: x = 2.0, dx = -0.5, cell_size = 2:
        i = 1 - 1 = 0
        t = (0 * cell_size - x) / dx = -2 / -0.5 = 4

    example 3: x = 7.5, dx = 0.5, cell_size = 1:
        i = 7
        t = (8 * cell_size - x) / dx = 0.5 / 0.5 = 1

    example 4: x = 7.78, dx = 0.5, cell_size = 1:
        i = 7
        t = (8 * cell_size - x) / dx = 0.22 / 0.5 = 0.44

    """
    t = 0.0

    i = int(x // cell_size)
    if i == x // cell_size:
        if dx < 0:
            i -= 1
    if dx > 0:
        t = ((i + 1) * cell_size - x) / dx
    elif dx < 0:
        t = (i * cell_size - x) / dx
    else:
        t = np.inf

    return t


@ti.func
def find_dt(x: float, y: float, dx: float, dy: float, max_level: int):
    assert dx != 0 or dy != 0
    step_size = 2**max_level
    tx = partial_dt(x, dx, step_size)
    ty = partial_dt(y, dy, step_size)
    return min(tx, ty)


@ti.func
def find_new_tangent(x: float, y: float, z: float, z0: float, dx: float, dy: float,
                     t: float, dt: float, tangent: float, tan1: float, height_field: ti.types.ndarray()):
    for i in range(10):
        t_sample = dt * i / 10.0
        x_sample = x + t_sample * dx
        y_sample = y + t_sample * dy
        z_sample = get_height(height_field, x_sample, y_sample)
        z_projection = z + t_sample * tangent
        if z_sample > z_projection:
            tangent = (z_sample - z0) / (t + t_sample)
            if tangent > tan1:
                break
    return tangent


@ti.func
def max_tangent(x: float, y: float, dx: float, dy: float, tan0: float, tan1: float,
                height_field: ti.types.ndarray(), maxmipmap: ti.types.ndarray(), n_levels: int):
    tangent = tan0
    w, h = height_field.shape
    z0 = get_height(height_field, x, y)
    z = z0
    t = 0.0
    dt = 0.0

    while 0 <= x < w - 1 and 0 <= y < h - 1 and tangent <= tan1:
        max_level = get_max_level(maxmipmap, x, y, z, dx, dy, n_levels)
        if max_level == n_levels:
            break
        elif 0 <= max_level < n_levels:
            dt = find_dt(x, y, dx, dy, max_level)
            if dt <= 0:
                break
        else:
            dt = find_dt(x, y, dx, dy, 0)
            if dt <= 0:
                break
            tangent = find_new_tangent(x, y, z, z0, dx, dy, t, dt, tangent, tan1, height_field)

        t += dt
        x += dt * dx
        y += dt * dy
        z += dt * tan0

    return tangent