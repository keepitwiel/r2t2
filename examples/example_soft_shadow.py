import numpy as np
import matplotlib.pyplot as plt

from r2t2.optimally_fast_soft_shadows.soft_shadow import fast_soft_shadow_algorithm, create_maxmipmap, shadow_fraction
from example1 import example_map_1


def main(n_cells: int) -> None:
    N_prime = 5
    n_nodes = n_cells + 1
    # height_field = np.random.uniform(size=(n_nodes, n_nodes)) # np.zeros((n_nodes, n_nodes))
    height_field, _ = example_map_1(n_nodes)
    maxmipmap = create_maxmipmap(height_field)
    J_star = np.zeros((n_cells, n_cells))
    alt = np.pi / 6
    azi = 0 * np.pi
    dr = np.array([np.cos(azi) * np.cos(alt), np.sin(azi) * np.cos(alt), np.sin(alt)])
    for i in range(n_cells):
        for j in range(n_cells):
            x = (i + 0.5) / n_cells
            y = (j + 0.5) / n_cells
            z = np.mean(height_field[i:i+2, j:j+2])

            r = np.array([x, y, z])
            J_star[i, j] = fast_soft_shadow_algorithm(N_prime, r, dr, maxmipmap)

    surface_normal = np.array([0, 0, 1])
    rho = np.sqrt(1 - dr[2]**2)
    ray_normal = np.array([-dr[2] * dr[0], -dr[2] * dr[1], rho])
    print(surface_normal.dot(ray_normal))
    shadow = shadow_fraction(J_star, surface_normal, ray_normal, light_radius=np.pi / 180)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(height_field, cmap="terrain")
    axes[1].imshow(np.maximum(0, J_star))
    axes[2].imshow(shadow)
    plt.imshow(J_star)

    plt.show()


if __name__ == "__main__":
    main(n_cells=64)
