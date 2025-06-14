import numpy as np
import matplotlib.pyplot as plt


"""
Table of symbols
================
p           View ray intersection point
L^          Shadow (light) ray unit vector
R_obj       Object space position vector of tip of ray
R_tex       Texture space position vector of tip of ray
h           Terrain height in [0,1] increasing from bottom to top
max_h       Maximum h within a maximum mipmap texel
H           Height of tip of ray
∆_max_h_k   Diﬀerence between max h and H
T           Length of ray covering one maximum mipmap texel
t           Distance traveled by shadow ray normalized to [0,T] →[0,1]
N           Height field size
N′          Total number of mipmap levels
m           Mip level
∆M          Size of mipmap texel
k           Current iteration step
J*          Optimal cost over {0,1,...,k}


Algorithm 1: Fast Soft Shadow Algorithm
=======================================
 1. J*0 ← 1
 2. t0 ← 1
 3. m ← N′ − 1
 4. ∆R ← Texel step size
 5. for k ← 0 to N′−1 do
 6.    t* ←−1
 7.    ∆_max_h* ←1
 8.    i ← 0
 9.    Compute height H of ray tip at R(t_{k,i})
10.    Sample max_h(t_{k+1}) at R(t_{k,i}), mip level m
11.    ∆_max_h_k ← H − max_h(t_{k+1})
12.    if ∆_max_h_k <∆_max_h* then
13.       ∆_max_h* ← ∆_max_h(t_{k+1})
14.       t* ← t_k
15.    for i ← 1 to 2 do // DDA
16.        R(t_{k,i}) ← R(t_{k,i−1}) − 2^(−k−1)∆R
17.        t_k ← t_k − 2^(−k−1)
18.        Compute height H of ray tip at R(t_{k,i})
19.        Sample max_h(t_{k+1}) at R(t_{k,i}), mip level m
20.        ∆_max_h_k ← H − max_h(t+{k+1})
21.        if ∆_max_h_k <∆_max_h* then
22.            ∆_max_h* ←∆_max_h_k
23.            t* ←t_k
24.    m ← m − 1, k ← k + 1
25.    if t* > −1 then
26.        t_k ← t* + 2^(−k)
27. if ∆_max_h* <1 then
28.    J* ←(∆_max_h*)/t*
"""


def create_maxmipmap(z: np.ndarray) -> np.ndarray:
    """
    Given height field z, compute maxmipmap
    """
    # First, calculate the maximum between the log of width and height.
    w, h = z.shape
    assert w == h
    n_cells = w - 1
    log_n_cells = np.log2(n_cells)
    n_levels = int(log_n_cells)
    assert n_levels == log_n_cells

    maxmipmap = -np.inf + np.zeros((n_cells * 2, n_cells))
    z_stack = np.stack([z[:-1, :-1], z[1:, :-1], z[:-1, 1:], z[1:, 1:]], axis=2)
    z_max = np.max(z_stack, axis=2)
    maxmipmap[:n_cells, :n_cells] = z_max

    offset = n_cells
    for _ in range(n_levels):
        n_cells //= 2
        z_stack = np.stack([z_max[::2, ::2], z_max[1::2, ::2], z_max[::2, 1::2], z_max[1::2, 1::2]], axis=2)
        z_max = np.max(z_stack, axis=2)
        maxmipmap[offset:offset + n_cells, :n_cells] = z_max
        offset += n_cells

    return maxmipmap


def sample_maxmipmap(ray_position: np.ndarray, maxmipmap: np.ndarray, m: int) -> float:
    i, j = int(ray_position[0]), int(ray_position[1])
    n_cells = maxmipmap.shape[1]
    step_size = 1
    offset = 0
    for _ in range(m):
        offset += n_cells // step_size
        step_size *= 2
        i //= 2
        j //= 2
    result = float(maxmipmap[offset + i, j])
    if result == -np.inf:
        print(f"wrong maxmipmap value for ray_position={ray_position}, level={m}")
    return float(maxmipmap[offset + i, j])



def fast_soft_shadow_algorithm(
    N_prime: int,
    delta_R: float,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    maxmipmap: np.ndarray
) -> float:
    """
    Fast Soft Shadow Algorithm.
    Args:
        N_prime: Number of steps (N′).
        delta_R: Texel step size (∆R).
        ray_origin: NumPy array, ray starting point. (3D)
        ray_direction: NumPy array, ray direction (normalized). (3D)
        maxmipmap: NumPy array, contains maximum mipmap values for the height field
    Returns:
        J_star: Shadow intensity (scalar).
    """
    J_star_0 = 1.0  # J*_0 ← 1
    t0 = 1.0  # t_0 ← 1
    m = N_prime - 1  # m ← N′ − 1
    t_k = t0
    J_star = J_star_0
    delta_max_h_star = 1.0
    t_star = -1.0

    for k in range(N_prime):  # for k ← 0 to N′−1 do
        t_star = -1.0  # t* ← −1
        delta_max_h_star = 1.0  # ∆_max_h* ← 1

        # Initial ray position at i = 0
        ray_position = ray_origin + t_k * ray_direction
        if not (0 <= ray_position[0] < 1 and 0 <= ray_position[1] < 1):
            break
        delta_max_h_k = ray_position[2] - sample_maxmipmap(ray_position, maxmipmap, m)  # ∆_max_h_k ← H − max_h(t_{k+1})
        if delta_max_h_k < delta_max_h_star:  # if ∆_max_h_k < ∆_max_h*
            delta_max_h_star = delta_max_h_k  # ∆_max_h* ← ∆_max_h_k
            t_star = t_k  # t* ← t_k

        for i in range(1, 3):  # for i ← 1 to 2 do // DDA
            step_size = 2 ** (-k - 1) * delta_R
            ray_position = ray_position - step_size * ray_direction  # R(t_{k,i}) ← R(t_{k,i−1}) − 2^(−k−1)∆R
            if not (0 <= ray_position[0] < 1 and 0 <= ray_position[1] < 1):
                break
            t_k = t_k - 2 ** (-k - 1)  # t_k ← t_k − 2^(−k−1)
            delta_max_h_k = ray_position[2] - sample_maxmipmap(ray_position, maxmipmap, m)  # ∆_max_h_k ← H − max_h(t_{k+1})
            if delta_max_h_k < delta_max_h_star:  # if ∆_max_h_k < ∆_max_h*
                delta_max_h_star = delta_max_h_k  # ∆_max_h* ← ∆_max_h_k
                t_star = t_k  # t* ← t_k

        m = m - 1  # m ← m − 1

        if t_star > -1:  # if t* > −1
            t_k = t_star + 2 ** (-k)  # t_k ← t* + 2^(−k)

    if delta_max_h_star < 1:  # if ∆_max_h* < 1
        J_star = delta_max_h_star / t_star if t_star != 0 else J_star_0  # J* ← (∆_max_h*)/t*

    return J_star


def main():
    N_prime = 5
    delta_R = 1.0
    n_cells = 16
    n_nodes = n_cells + 1
    height_field = np.zeros((n_nodes, n_nodes))
    maxmipmap = create_maxmipmap(height_field)
    J_star = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            x = (i + 0.5) / n_cells
            y = (j + 0.5) / n_cells
            r = np.array([x, y, maxmipmap[i, j]])
            dr = np.array([0.5, 0.5, np.sqrt(0.5)])
            J_star[i, j] = fast_soft_shadow_algorithm(N_prime, delta_R, r, dr, maxmipmap)
            print(i, j, r, dr, J_star[i, j])

    plt.imshow(J_star)
    plt.show()


if __name__ == "__main__":
    main()
