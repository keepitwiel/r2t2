import taichi as ti
import numpy as np


@ti.func
def get_propagation_length(
        r: ti.math.vec3, dr: ti.math.vec3, maxmipmap: ti.types.ndarray(), height_field: ti.types.ndarray()):
    """
    Given
    - a position r and direction dr in ray traversion space

    - determine a bounding box around (r.x, r.y) depending on
      which maxmipmap level we are (determined by r.z);
    - find a point (x', y') = (x + l * dx, y + l * dy)
      on the bounding box that is closest to (x, y);
    - return l.

    This is the guaranteed length a ray can travel without
    colliding with the height map.
    """
    result = 0.0
    step = 1
    offset = 0
    k = maxmipmap.shape[0]
    level = 0
    while step < maxmipmap.shape[0]:
        i = get_hierarchical_index(r.x, dr.x, step)
        j = get_hierarchical_index(r.y, dr.y, step)
        z_max = get_height(height_field, i, j) if level == 0 else maxmipmap[i, offset + j]
        if r.z < z_max:
            step = step // 2
            break
        offset += k
        step *= 2
        k = k // 2
        level += 1

    if step >= 1:
        i = get_hierarchical_index(r.x, dr.x, step)
        j = get_hierarchical_index(r.y, dr.y, step)
        l_left = length_to_boundary(r.x, i * step, dr.x)
        l_right = length_to_boundary(r.x, (i + 1) * step, dr.x)
        l_top = length_to_boundary(r.y, j * step, dr.y)
        l_bottom = length_to_boundary(r.y, (j + 1) * step, dr.y)
        result = min(l_left, l_right, l_top, l_bottom)

    return result


@ti.func
def get_hierarchical_index(x: float, dx: float, step: float):
    """
    returns the divisor of x. If x is a multiple of step, and the
    direction of travel is negative (dx < 0), substract one.

    Example 1: x = 32.5, dx = -0.3, step = 8
    Returns: int(32.5 // 8) - 0 = 4

    Example 2: x = 16.0, dx = 0.5, step = 4
    Returns: int(16.0 // 4) - 0 = 4

    Example 3: x = 16.0, dx = -0.5, step = 4
    Returns: int(16.0 // 4) - 1 = 3
    """
    return int(x // step) - (1 if x % step == 0.0 and dx < 0 else 0)


@ti.func
def length_to_boundary(x: float, x_boundary: float, dx: float):
    result = np.inf
    if dx != 0:
        result = (x_boundary - x) / dx
        result = result if result > 0 else np.inf
    return result


@ti.func
def get_height(height_field: ti.types.ndarray(), x: float, y: float):
    int_x = int(x)
    int_y = int(y)
    z = 0.25 * (
        height_field[int_x, int_y] +
        height_field[int_x + 1, int_y] +
        height_field[int_x, int_y + 1] +
        height_field[int_x + 1, int_y + 1]
    )  # simplified height - average of four corners. should be precomputed or use interpolation
    return z


@ti.func
def spherical_to_euclidean(azimuth: float, altitude: float):
    a = ti.cos(altitude)
    dr = ti.math.vec3(a * ti.cos(azimuth), a * ti.sin(azimuth), ti.sin(altitude))
    return dr


@ti.func
def get_propagation_length_2(r: ti.math.vec2, dr: ti.math.vec2, step_size: int):
    # get tx
    x_cell = r.x - (r.x % step_size)
    xmin, xmax = x_cell, x_cell
    if dr.x >= 0:
        xmax += step_size
    else:
        xmin -= step_size
    tx = np.inf
    if dr.x != 0.0:
        if dr.x > 0.0:
            tx = (xmax - r.x) / dr.x
        else:
            tx = (xmin - r.x) / dr.x

    # get ty
    y_cell = r.y - (r.y % step_size)
    ymin, ymax = y_cell, y_cell
    if dr.y >= 0:
        ymax += step_size
    else:
        ymin -= step_size
    ty = np.inf
    if dr.y != 0.0:
        if dr.y > 0.0:
            ty = (ymax - r.y) / dr.y
        else:
            ty = (ymin - r.y) / dr.y

    # find minimum step length
    t_min = min(tx, ty)
    return t_min

@ti.func
def sample_altitude_simple(
    x: float,
    y: float,
    azimuth: float,
    min_altitude: float,
    max_altitude: float,
    height_field: ti.types.ndarray(),
):
    # setup
    n_cells = height_field.shape[0] - 1
    theta = min_altitude
    r0 = ti.math.vec2(x, y)
    z0 = get_height(height_field, x, y)
    r = r0
    dr = ti.math.vec2(ti.cos(azimuth), ti.sin(azimuth))
    step_size = 1

    while 0 <= r.x < n_cells and 0 <= r.y < n_cells and theta <= max_altitude:
        t_min = get_propagation_length_2(r, dr, step_size)
        r += t_min * dr
        z_sample = get_height(height_field, r.x, r.y)
        w = r - r0
        theta_sample = ti.atan2(z_sample - z0, ti.sqrt(w.x * w.x + w.y * w.y))
        if theta < theta_sample:
            theta = theta_sample

    if theta >= max_altitude:
        theta = max_altitude
    return theta



@ti.func
def sample_altitude(
    x: float,
    y: float,
    azimuth: float,
    min_altitude: float,
    max_altitude: float,
    maxmipmap: ti.types.ndarray(),
    height_field: ti.types.ndarray(),
):
    """
    Propagates a ray from (x, y, z), with z = height field(x, y) and
    direction (azimuth, min_altitude)
    If no collisions with terrain occur, returns min_altitude from input.
    Otherwise, raises min_altitude with each collision such that z = height_field(x, y)
    and keeps propagating
    """
    t = 0.0
    result = min_altitude
    field_size = maxmipmap.shape[1]
    z = get_height(height_field, x, y)
    r0 = ti.math.vec3(x, y, z)
    r = ti.math.vec3(x, y, z)

    dr = spherical_to_euclidean(azimuth, min_altitude)
    # print(f"start propagation: r={r}, dr={dr}, min_altitude={min_altitude:.2f}")


    while True:
        dt = get_propagation_length(r, dr, maxmipmap, height_field)
        if dt == 0.0:
            if azimuth >= np.pi / 2:
                print(azimuth, dt, r, dr)
            # ray collided with terrain - raise r.z and continue propagation
            r.z = get_height(height_field, r.x, r.y)
            diff = r - r0
            result = ti.atan2(diff.z, ti.sqrt(diff.x * diff.x + diff.y * diff.y))
            # print(f"collision, raising z to {r.z}, raising min_altitude to {result}")
            if result >= max_altitude:
                result = max_altitude
                # print(f"min_altitude before break: {result:.2f}")
                break

            # todo: turn into function
            dr = spherical_to_euclidean(azimuth, result)
        else:
            t += dt
            r += dt * dr
        if r.x <= 0.0 or r.x >= field_size or r.y <= 0.0 or r.y >= field_size:
            # print(f"min_altitude before break: {result:.2f}")
            break

    if result < min_altitude:
        result = min_altitude
    # print(f"min_altitude before/after propagation: {min_altitude:.2f}/{result:.2f}")
    return result