import numpy as np

QUARTER_PI = np.pi / 4

MATRICES = np.array(
    [
        [
            [1, 0],
            [0, 1],
        ],
        [
            [0, 1],
            [1, 0],
        ],
        [
            [0, 1],
            [-1, 0],
        ],
        [
            [-1, 0],
            [0, 1],
        ],
        [
            [-1, 0],
            [0, -1],
        ],
        [
            [0, -1],
            [-1, 0],
        ],
        [
            [0, -1],
            [1, 0],
        ],
        [
            [1, 0],
            [0, -1],
        ]
    ]
)


def line_in_octant(theta: float, n_iterations: int) -> tuple[np.ndarray, float]:
    assert 0 <= theta <= QUARTER_PI
    line_coordinates = np.zeros((n_iterations, 2, 2), dtype=int)
    k = np.arange(1, n_iterations)
    delta = np.tan(theta)
    projection_length = np.sqrt(1 + delta * delta)
    y = k * delta
    y_floor = y.astype(int)
    offset = 1 if theta % QUARTER_PI == 0 else 0
    line_coordinates[1:, 0, 0] = k
    line_coordinates[1:, 0, 1] = y_floor
    line_coordinates[1:, 1, 0] = k
    line_coordinates[1:, 1, 1] = y_floor + offset
    return line_coordinates, projection_length


def line(theta: float, n_iterations: int) -> tuple[np.ndarray, float]:
    theta_ = theta % QUARTER_PI
    m = int(theta / QUARTER_PI)  # octant number
    if m % 2 == 1:
        theta_ = QUARTER_PI - theta_
    octant_coordinates, projection_length = line_in_octant(theta_, n_iterations)
    transformation_matrix = MATRICES[m]
    line_coordinates = octant_coordinates @ transformation_matrix
    return line_coordinates, projection_length
