import taichi as ti


@ti.kernel
def illuminate(
        minmipmap: ti.types.ndarray(),
        maxmipmap: ti.types.ndarray(),
        lb_altitude: ti.types.ndarray(),
        azimuth: float,
        min_altitude: float,
        max_altitude: float,
        out_array: ti.types.ndarray(),
):
    pass