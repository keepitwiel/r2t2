import taichi as ti
import numpy as np


@ti.func
def get_height(height_field: ti.types.ndarray(), r: ti.math.vec2):
    """
    Given a height field, which is a regular grid of nodes
    with height values on the domain [0, W] x [0, H],
    determine the height at vector r. The height is given
    as a bilinear interpolation function of r.

    If r is outside the domain, return negative infinity.
    """
    w, h = height_field.shape
    i, j = int(r[0]), int(r[1])
    result = -np.inf
    if 0 <= i < w - 1and 0 <= j < h - 1:
        lx, ly = r[0] - i, r[1] - j
        result = (
            (1 - lx) * (1 - ly) * height_field[i, j] +
            lx * (1 - ly) * height_field[i + 1, j] +
            (1 - lx) * ly * height_field[i, j + 1] +
            lx * ly * height_field[i + 1, j + 1]
        )
    return result


@ti.func
def max_tangent(
    height_field: ti.types.ndarray(),
    log_min: int,
    log_max: int,
    r: ti.math.vec2,
    dr:ti.math.vec2,
    z: float,
    tan: float,
    tan_upper_bound: float,
):
    """
    For a 2D ray r with direction dr, loop
    through values of t and find the maximum tangent
    between the height at r + t * dr and t
    as long as tangent remains below tan_upper_bound.

    This method works well for smooth terrains;
    For rough terrains, either smaller increments
    should be chosen for t or some kind of error
    correction should be implemented where
    tangents are compared against each other.

    if radius = 0,
    :param height_field:
    :param log_min:
    :param log_max:
    :param r: ray (x, y) coordinates
    :param dr: ray direction
    :param z: starting height of ray
    :param tan: starting tangent
    :param tan_upper_bound: if tangent gets above this value, we can stop
    :return: maximum tangent
    """
    tan_max = tan
    t = 0
    for k in range(log_min, log_max):
        t += 2**k * ti.random(float)
        r_sample = r + t * dr
        z_sample = get_height(height_field, r_sample)
        if z_sample > -np.inf:
            tan = (z_sample - z) / t
            if tan > tan_max:
                tan_max = tan
            if tan_max > tan_upper_bound:
                break
        else:
            break
    return tan_max


@ti.kernel
def render_maximum_altitude(
    output_array: ti.types.ndarray(),
    height_field: ti.types.ndarray(),
    log_min: int,
    log_max: int,
    azi: float,
    alt: float,
    radius: float,
):
    """
    For each cell in the output array, determine the illumination.
    1.0 = fully lit, 0.0 = fully shadowed
    """
    tan_upper_bound = alt + radius
    tan = ti.tan(alt)
    dr = ti.math.vec2(ti.cos(azi), ti.sin(azi))
    for i, j in output_array:
        r = ti.math.vec2(i + ti.random(float), j + ti.random(float))
        z = height_field[i, j]
        tangent = max_tangent(
            height_field, log_min, log_max, r, dr, z, tan=tan, tan_upper_bound=tan_upper_bound
        )
        if radius > 0:
            output_array[i, j] = (tan_upper_bound - ti.atan2(tangent, 1)) / radius
        else:
            if alt >= tangent:
                output_array[i, j] = 1.0
            else:
                output_array[i, j] = 0.0
