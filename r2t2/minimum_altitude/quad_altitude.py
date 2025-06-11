import taichi as ti

from .sample_altitude import sample_altitude


EPSILON = 0.01


@ti.func
def mean(buffer: ti.types.ndarray()):
    result = 0.0
    n = buffer.shape[0]
    for k in range(n):
        result += buffer[k]
    return result / n


@ti.func
def error_min_max(
    buffer: ti.types.ndarray(),
    min_value: float,
    max_value: float,
) -> float:
    n_samples = buffer.shape[0]
    for i in range(n_samples):
        if buffer[i] < min_value:
            min_value = buffer[i]
        if buffer[i] > max_value:
            max_value = buffer[i]
    diff = max_value - min_value
    return diff


@ti.kernel
def quad_altitude_sampling(
    height_field: ti.types.ndarray(),
    maxmipmap: ti.types.ndarray(),
    azimuth: float,
    global_min_altitude: float,
    global_max_altitude: float,
    n_levels: int,
    n_samples: int,
    buffer: ti.types.ndarray(),
    out_array: ti.types.ndarray(),
):
    """
    top_level_cell = min_alt
    - per level (top -> bottom):
        - per cell:
            - ray_alts = []
            - for k in range(N):
                - ray_alt = sample_ray_altitude(random_x, random_y, min_alt)
                - ray_alts.append(ray_alt)
            - noise calculate_noise(ray_alts)
            - if is_acceptible(noise):
                set_flag_for_lower_levels()
            - set lower level cells to estimate_minimum(ray_alts)


    """
    print(azimuth)
    dimension = 2**n_levels
    for inv in range(n_levels):
        level = n_levels - inv - 1
        step_size = 2**level
        grid_size = dimension // step_size

        for i in range(grid_size):
            for j in range(grid_size):
                # print(f"level={level}, i={i}, j={j}, step_size={step_size}, grid_size={grid_size}, top level size={dimension}")
                if out_array[i * step_size, j * step_size] == global_min_altitude:
                    # this region has not been touched; sample and test
                    # if the sample mean is accurate enough.
                    # if not, we will leave this region untouched
                    # so that it can be visited in the next
                    for k in range(n_samples):
                        x = step_size * (i + ti.random(dtype=float))
                        y = step_size * (j + ti.random(dtype=float))
                        buffer[k] = sample_altitude(x, y, azimuth, global_min_altitude, global_max_altitude, maxmipmap,
                                                    height_field)
                    error = error_min_max(buffer, global_min_altitude, global_max_altitude)
                    if error < EPSILON or step_size == 1:
                        avg = mean(buffer)
                        # print(f"mean of samples has error {error:.2f} with step size {step_size}, we set this region"
                        #       f" ({i*step_size}:{(i+1)*step_size}, {j*step_size}:{(j+1)*step_size}) to {avg:.2f}")
                        # set out_array to avg
                        for u in range(step_size):
                            for v in range(step_size):
                                out_array[i * step_size + u, j * step_size + v] = avg
                # else:
                #     print(f"this region ({i*step_size}:{(i+1)*step_size}, {j*step_size}:{(j+1)*step_size}) has been touched, skipping...")