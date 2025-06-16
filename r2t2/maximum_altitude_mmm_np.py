import numpy as np


def reduce(array: np.ndarray, step_size: int, is_max: bool) -> np.ndarray:
    if is_max:
        func = np.max
    else:
        func = np.min
    result = func(
        np.stack(
            [
                array[:-1:step_size, :-1:step_size],
                array[1::step_size, :-1:step_size],
                array[:-1:step_size, 1::step_size],
                array[1::step_size, 1::step_size],
            ],
            axis=2,
        ),
        axis=2,
    )
    return result


def get_mipmap(z: np.ndarray, is_max: bool) -> list[np.ndarray]:
    w, h = z.shape
    assert w == h
    n_levels = int(np.log2(w - 1))
    buffer = reduce(z, step_size=1, is_max=is_max)
    result = [buffer]
    for level in range(n_levels):
        buffer = reduce(buffer, step_size=2, is_max=is_max)
        result.append(buffer)
    return result


def get_height(height_field: np.ndarray, x: float, y: float):
    w, h = height_field.shape
    if 0 <= x < w - 1 and 0 <= y < h - 1:
        i, j = int(x), int(y)
        u, v = x % 1, y % 1
        z00 = height_field[i, j]
        z10 = height_field[i + 1, j]
        z01 = height_field[i, j + 1]
        z11 = height_field[i + 1, j + 1]
        return (1 - u) * (1 - v) * z00 + u * (1 - v) * z10 + (1 - u) * v * z01 + u * v * z11
    else:
        return -np.inf


def get_max_level(maxmipmap, x, y, z, dx, dy):
    n_levels = len(maxmipmap)
    step_size = 1
    for level in range(n_levels):
        i, j = int(x // step_size), int(y // step_size)
        if i == x:
            if dx < 0:
                i -= 1
        if j == y:
            if dy < 0:
                j -= 1
        w, h = maxmipmap[level].shape
        if 0 <= i < w and 0 <= j < h:
            z_max = maxmipmap[level][i, j]
            if z < z_max:
                level -= 1
                break
        else:
            # we're outside the domain - we are above the terrain by definition
            return n_levels
        step_size *= 2
    return level


def partial_dt(x, dx, cell_size):
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


def find_dt(x, y, dx, dy, max_level):
    assert dx != 0 or dy != 0
    step_size = 2**max_level
    tx = partial_dt(x, dx, step_size)
    ty = partial_dt(y, dy, step_size)
    return min(tx, ty)


def find_new_tangent(x, y, z, z0, dx, dy, t, dt, tangent, height_field):
    for t_sample in np.linspace(0, dt, 10):
        x_sample = x + t_sample * dx
        y_sample = y + t_sample * dy
        z_sample = get_height(height_field, x_sample, y_sample)
        z_projection = z + t_sample * tangent
        if z_sample > z_projection:
            tangent = (z_sample - z0) / (t + t_sample)
    return tangent


def max_tangent(x, y, dx, dy, tangent, height_field, maxmipmap):
    n_levels = len(maxmipmap)
    w, h = height_field.shape
    z0 = get_height(height_field, x, y)
    z = z0
    t = 0

    while 0 <= x < w - 1 and 0 <= y < h - 1:
        max_level = get_max_level(maxmipmap, x, y, z, dx, dy)
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
            tangent = find_new_tangent(x, y, z, z0, dx, dy, t, dt, tangent, height_field)

        t += dt
        x += dt * dx
        y += dt * dy
        z += dt * tangent

    return tangent