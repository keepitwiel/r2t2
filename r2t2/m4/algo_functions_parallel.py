import numpy as np
import taichi as ti


@ti.func
def test_if_blocked(
    min_array: ti.types.ndarray(),
    max_array: ti.types.ndarray(),
    line_coordinates: ti.types.ndarray(),
    dz: float,
    i: int,
    j: int,
    global_max: float,
) -> bool:
    """
    Returns True if reference cell at (i, j) is "blocked"
    by two cells at any iteration along a thick line,
    given by line_coordinates.
    """
    result = False
    w, h = min_array.shape
    z_projected = max_array[i, j]  # initially, projected height is the maximum of the current cell
    n = line_coordinates.shape[0]
    for k in range(1, n):
        z_projected += dz
        if z_projected > global_max:
            # projected height is above global maximum - we can stop
            break

        # get test cell coordinates
        i0 = i + line_coordinates[k, 0, 0]
        j0 = j + line_coordinates[k, 0, 1]
        i1 = i + line_coordinates[k, 1, 0]
        j1 = j + line_coordinates[k, 1, 1]

        inside0 = 0 <= i0 < w and 0 <= j0 < h
        inside1 = 0 <= i1 < w and 0 <= j1 < h

        if not (inside0 or inside1):
            # we have left the pixel space
            break

        # get test cell values
        z_min0 = min_array[i0, j0] if inside0 else -np.inf
        z_min1 = min_array[i1, j1] if inside1 else -np.inf

        if z_projected < z_min0 and z_projected < z_min1:
            # projected height is smaller than both test minima,
            # meaning our reference cell is blocked at least once
            # over the course of the line.
            result = True
            break

    return result


@ti.kernel
def fill_shadow_array(
        array: ti.types.ndarray(),
        previous_array: ti.types.ndarray(),
        min_array: ti.types.ndarray(),
        max_array: ti.types.ndarray(),
        line_coordinates: ti.types.ndarray(),
        dz: float,
        global_max: float,
):
    for i_parent, j_parent in previous_array:
        # first, look up parent cell
        parent_value = previous_array[i_parent, j_parent]
        i, j = i_parent * 2, j_parent * 2
        if parent_value:
            # parent cell was blocked, so by definition
            # all child cells are blocked
            for i_test in range(i, i + 2):
                for j_test in range(j, j + 2):
                    array[i_test, j_test] = True
        else:
            # we are not sure that the parent cell was
            # completely blocked, so we loop over child cells
            for i_test in range(i, i + 2):
                for j_test in range(j, j + 2):
                    array[i_test, j_test] = test_if_blocked(
                        min_array=min_array,
                        max_array=max_array,
                        line_coordinates=line_coordinates,
                        dz=dz,
                        i=i_test,
                        j=j_test,
                        global_max=global_max,
                    )
