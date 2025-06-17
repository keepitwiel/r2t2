import numpy as np
import taichi as ti


@ti.kernel
def render(
    output_field: ti.types.ndarray(),
    height_field: ti.types.ndarray(),
    maxmipmap: ti.types.ndarray(),
    azi: float,
    alt: float,
    diameter: float,
    n_levels: int,
):
    """
    Render an illumination field based on various inputs.

    params:
        output_field: illumination field indicating how much
            light falls on a cell bounded by four grid nodes.
            0.0 = shadow, 1.0 = fully lit
        height_field: height values of grid nodes on a square, regular grid.
            size of height_field = size of output_field + 1
        maxmipmap: a maximum mipmap of the height_field
        azi: azimuth of light source
        alt: lower bound altitude of light source
        diameter: diameter of light source
        n_levels: number of levels in maxmipmap

    returns:
        None
    """
    dr = ti.math.vec2(ti.cos(azi), ti.sin(azi))
    tan0 = ti.tan(alt)
    tan1 = ti.tan(alt + diameter)
    for i, j in output_field:
        r = ti.math.vec2(i + 0.5, j + 0.5)
        tangent = max_tangent(r, dr, tan0, tan1, height_field, maxmipmap, n_levels)
        if diameter > 0:
            tangent = min(tangent, tan1)  # have to clip
            output_field[i, j] = (ti.atan2(tan1, 1) - ti.atan2(tangent, 1)) / diameter
        else:
            """
            in the case of a point light source (diameter = 0),
            the lower bound and upper bound altitudes are the same (tan0 == tan1).
            Consequently, a point on the height field is in shadow if
            tan1 < tangent (light source tangent is above the maximum tangent;
            if tan1 == tangent, no collision with height field has taken place
            and therefore the point is fully lit
            """
            if tan1 == tangent:
                output_field[i, j] = 1.0
            else:
                output_field[i, j] = 0.0


@ti.kernel
def one_step_mipmap(inp: ti.types.ndarray(), out: ti.types.ndarray()):
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


@ti.kernel
def mipmap_first_level(inp: ti.types.ndarray(), out: ti.types.ndarray()):
    n = out.shape[1]
    ti.loop_config(block_dim=16)
    for i, j in ti.ndrange(n, n):
        out[i, j] = max(inp[i, j], inp[i + 1, j], inp[i, j + 1], inp[i + 1, j + 1])

@ti.kernel
def mipmap_level(inp: ti.types.ndarray(), out: ti.types.ndarray(), source_offset: int, target_offset: int, m: int):
    ti.loop_config(block_dim=16)
    for i, j in ti.ndrange(m, m):
        out[i + target_offset, j] = max(
            inp[i * 2 + source_offset, j * 2],
            inp[i * 2 + source_offset + 1, j * 2],
            inp[i * 2 + source_offset, j * 2 + 1],
            inp[i * 2 + source_offset + 1, j * 2 + 1],
        )


def compute_mipmap(inp: np.ndarray, out: np.ndarray):
    """
    Compute mipmap by outsourcing the two steps to different kernels.
    Should be faster for GPUs.
    """
    w, h = inp.shape
    print(w, h)
    n2, n = out.shape
    print(n2, n)
    assert w == h, "input width and height have to be the same"
    assert (n + 1) * 2 == n2, "output array has to have width = double height"
    assert n == h - 2, "output array height has to be same as input height minus one"

    mipmap_first_level(inp, out)
    step = 2
    source_offset = 0
    target_offset = n
    while step <= n:
        m = n // step
        mipmap_level(out, out, source_offset, target_offset, m)
        source_offset = target_offset
        target_offset += m
        step *= 2


@ti.func
def get_height(
    height_field: ti.types.ndarray(),
    r: ti.math.vec2,
):
    """
    Calculate f(x, y), where f is a bilinear interpolation of the gridpoints
    in height_field nearest to coordinate (x, y).
    """
    w, h = height_field.shape
    result = -np.inf
    if 0 <= r.x < w - 1 and 0 <= r.y < h - 1:
        i, j = int(r.x), int(r.y)
        u, v = r.x % 1, r.y % 1
        z00 = height_field[i, j]
        z10 = height_field[i + 1, j]
        z01 = height_field[i, j + 1]
        z11 = height_field[i + 1, j + 1]
        result = (1 - u) * (1 - v) * z00 + u * (1 - v) * z10 + (1 - u) * v * z01 + u * v * z11
    return result



@ti.func
def get_max_level(
    maxmipmap: ti.types.ndarray(),
    r: ti.math.vec2,
    z: float,
    dr: ti.math.vec2,
    n_levels: int,
):
    """
    Get height level at which ray height z is above the associated
    value in the maxmipmap.
    """
    step_size = 1
    l_ = 0
    offset = 0
    for l in range(n_levels):
        l_ = l
        d = 2**(n_levels - l)
        u, v = r.x // step_size, r.y // step_size
        i, j = int(u), int(v)
        if i == u and  dr.x < 0:
            i -= 1
        if j == v and dr.y < 0:
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
    Compute the smallest t > 0 such that x + t * dx is a multiple of cell_size.

    Args:
        x: Starting position (float).
        dx: Step direction and magnitude (float).
        cell_size: Grid cell size (int).

    Returns:
        Smallest t > 0, or ti.f32(np.inf) if dx is zero.

    Examples:
        partial_dt(1.0, -0.5, 1) -> 2.0
        partial_dt(2.0, -0.5, 2) -> 4.0
        partial_dt(7.5, 0.5, 1) -> 1.0
        partial_dt(7.78, 0.5, 1) -> 0.44
    """
    t = np.inf
    if dx != 0:
        i = ti.floor(x / cell_size, int)
        if dx < 0:
            i -= 1
        next_boundary = (i + 1) * cell_size
        t = (next_boundary - x) / dx
    return t


@ti.func
def find_dt(r: ti.math.vec2, dr: ti.math.vec2, max_level: int):
    """
    Find dt such that point (x + dt * dx, y + dt * dy) is at the closest
    grid edge, defined by max_level.
    """
    assert dr.x != 0 or dr.y != 0
    step_size = 2**max_level
    tx = partial_dt(r.x, dr.x, step_size)
    ty = partial_dt(r.y, dr.y, step_size)
    return min(tx, ty)


@ti.func
def find_new_tangent(
    r0: ti.math.vec2,
    z0: float,
    dr: ti.math.vec2,
    t: float,
    dt: float,
    tangent: float,
    tan1: float,
    height_field: ti.types.ndarray(),
):
    """
    Given a ray starting position (x0, y0, z0), ray length t, and tangent,
    determine if the ray intersects with the height field (by uniform stepping).
    If an intersection takes place, raise the tangent accordingly.
    """
    for i in range(1, 11):
        ts = t + dt * i / 10.0
        r_sample = r0 + ts * dr
        z_sample = get_height(height_field, r_sample)
        z_projection = z0 + ts * tangent
        if z_sample > z_projection:
            tangent = (z_sample - z0) / ts
            if tangent > tan1:
                break
    return tangent


@ti.func
def max_tangent(
    r: ti.math.vec2,
    dr: ti.math.vec2,
    tan0: float,
    tan1: float,
    height_field: ti.types.ndarray(),
    maxmipmap: ti.types.ndarray(),
    n_levels: int,
):
    """
    For a given ray at (x, y) and direction (dx, dy), find the lowest vertical tangent
    such that the ray doesn't intersect with the height field.
    """
    tangent = tan0
    w, h = height_field.shape
    r0 = r
    z0 = get_height(height_field, r)
    z = z0
    t = 0.0
    dt = 0.0

    while 0 <= r.x < w - 1 and 0 <= r.y < h - 1 and tangent <= tan1:
        max_level = get_max_level(maxmipmap, r, z, dr, n_levels)
        if max_level == n_levels:
            break
        else:
            dt = find_dt(r, dr, max(0, max_level))
        if max_level < 0:
            tangent = find_new_tangent(r0, z0, dr, t, dt, tangent, tan1, height_field)

        t += dt
        r += dt * dr
        z += dt * tan0

    return tangent