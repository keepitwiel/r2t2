import taichi as ti

ti.init(arch=ti.cpu)

import numpy as np


EPSILON = 1e-9
NEAR_INFINITE = 1e10


@ti.data_oriented
class SimpleReliefMapper:
    def __init__(self, height_map: np.ndarray, cell_size=1.0):
        self.w, self.h = height_map.shape
        self.altitude = np.pi / 2
        self.height_map = ti.field(dtype=float, shape=(self.w, self.h))
        self.height_map.from_numpy(height_map)
        self.max_value = np.max(height_map)
        self.min_value = np.min(height_map)
        self.pixels = ti.field(dtype=float, shape=(self.w, self.h))
        self.cell_size = cell_size
        self.maximum_ray_length = NEAR_INFINITE
        self.height_map_copy, self.maxmipmap, self.n_levels = self.get_maxmipmap()
        self.fill_maxmipmap()

    def get_maxmipmap(self):
        # calculate maximum mipmap using the height map as source image.
        #
        # If the source image does not have width/height that are of
        # the form 2**n, we force it to be so for convenience purposes.
        # this means filling the mipmap at lower levels with low values.
        #
        # First, calculate the maximum between the log of width and height.
        print("Get maxmipmap...")
        w, h = self.w, self.h
        log_w = np.log2(w)
        log_h = np.log2(h)
        max_log = max(log_w, log_h)

        # if the log is not an integer, we round up - this is the number of levels
        n_levels = int(max_log) + (1 if max_log != int(max_log) else 0)
        # print(n_levels)

        # Then, calculate the dimension parameter of the mipmap.
        dim = int(2**n_levels)

        # we copy the original image into a larger image
        height_map = ti.field(
            shape=(dim, dim),
            dtype=float
        )
        height_map.fill(-np.inf)

        # we create a mipmap field using the dimension parameter
        result = ti.field(
            shape=(dim // 2, dim - 1),
            dtype=float
        )
        result.fill(-np.inf)

        return height_map, result, n_levels

    @ti.kernel
    def fill_maxmipmap(self):
        w, h = self.w, self.h
        dim = int(2**self.n_levels)

        # fill the copy with original values
        print("fill the copy with original values...")
        for i in range(w):
            for j in range(h):
                self.height_map_copy[i, j] = self.height_map[i, j]

        print("Starting loops for maxmipmap...")

        y_source = 0
        y_target = 0
        dim_ = dim
        for level in range(self.n_levels):
            dim_ = dim_ // 2
            # print(level, dim_)
            # loop over new width and height to calculate the maximum of a window
            for i in range(dim_):
                for j in range(dim_):
                    # Define the window for max calculation
                    i0 = i * 2
                    j0 = y_source + j * 2
                    i1 = (i + 1) * 2
                    j1 = y_source + (j + 1) * 2

                    max_value = self.min_value
                    # Calculate maximum over the window
                    for u in range(i0, i1):
                        for v in range(j0, j1):
                            if level == 0:
                                # initially, we calculate the maximum of the height map over the window
                                max_value = ti.max(self.height_map[u, v], max_value)
                            else:
                                # once we have seeded the mipmap field, we can use it recursively
                                max_value = ti.max(self.maxmipmap[u, v], max_value)

                    self.maxmipmap[i, y_target + j] = max_value

            y_source = y_target
            y_target += dim_

        print("Done.")

    @ti.func
    def get_mipmap_value(self, i: int, j: int, level: int) -> float:
        n = self.n_levels
        # print(f"Number of levels: {n}, level requested: {level}")
        result = -1.0  # fallback value
        if 0 <= i < self.w and 0 <= j < self.h:
            if level == 0:
                result = float(self.height_map[i, j])
            elif level <= n:
                offset = 0
                for l in range(1, level):
                    dim = 2**(n - l)
                    offset += dim
                step = 2 ** level
                i_ = int(i // step)
                j_ = int(j // step)
                # print(f"level: {level}, step: {step}, offset: {offset}, dim: {dim}, i: {i}, j: {j}, i_: {i_}, j_: {j_}")
                result = float(self.maxmipmap[i_, offset + j_])
        return result

    @ti.kernel
    def get_mipmap_value_python_scope(self, i: int, j: int, level: int) -> float:
        return self.get_mipmap_value(i, j, level)

    @ti.func
    def get_partial_step_size(self, d: float, x: float, k: int) -> float:
        """
        Given a direction d, position x and magnitude k, find the length of the line
        between a boundary (defined by x and scale 2**k) and x.

        :param d:
        :param x:
        :param k:
        :return:
        """
        result = 0.0
        step = 2 ** k
        index = float(x // step)
        if d < 0.0:
            lb = index * step
            if x % step == 0.0:
                lb -= step
            result = (x - lb) / -d
        elif d > 0.0:
            ub = (index + 1.0) * step
            result = (ub - x) / d
        elif d == 0.0:
            result = self.maximum_ray_length
        assert result > 0
        return result

    @ti.func
    def max_test(self, x, y, z):
        """Find the highest maxmipmap level where z > the associated z_max for that level at (x, y)"""
        i, j = int(x), int(y)
        z_max = float(self.min_value)  # fallback
        final_level = 0  # fallback
        # for level in range(0, self.n_levels + 1):
        for level in range(self.n_levels + 1):
            level_ = self.n_levels - level
            z_max = self.get_mipmap_value(i, j, level=level_)
            if z_max < z:
                final_level = level_
                break
        return z_max, final_level

    @ti.func
    def get_step_size_to_next_bbox(
            self, x: float, y: float, z: float, dx: float, dy: float, classic: bool = True
    ) -> float:
        """
        Calculates the length a ray must travel before it hits
        the next relevant bounding box.

        The length, in part, depends on whether the ray is above
        the maximum of the surrounding height map.
        :param x:
        :param y:
        :param z:
        :param dx:
        :param dy:
        :param classic:
        :return:
        """
        result = 0.0
        if classic:
            # "classic" mode: the ray traverses the scene one cell at a time
            i, j = int(x), int(y)
            if self.height_map[i, j] > z:
                result = 0.0
            else:
                lx = self.get_partial_step_size(dx, x, 0)
                ly = self.get_partial_step_size(dy, y, 0)
                result = min(lx, ly)
                result += EPSILON
        else:
            # "mipmap" mode: the ray can skip cells if it's above all
            # cells in a given window at a hierarchical magnitude k

            # ==============================================================#
            # first, find the maximum z_max and hierarchical magnitude k    #
            # ==============================================================#
            z_max, k = self.max_test(x, y, z)
            if z_max > z and k == 0:
                #===========================================================#
                # even at the smallest magnitude,                           #
                # the maximum height is above z.                            #
                # therefore we conclude the ray is stopped by the terrain.  #
                #===========================================================#
                result = 0.0
            else:
                # TODO: duplicate code, fix
                lx = self.get_partial_step_size(dx, x, k)
                ly = self.get_partial_step_size(dy, y, k)
                result = min(lx, ly)
                result += EPSILON
        return result

    @ti.func
    def trace(self, i, j, dx, dy, dz, classic):
        result = 0.0
        w, h = self.height_map.shape

        #===============================================================#
        # within cell (i, j), randomly pick an (x, y) coordinate.       #
        # the z coordinate is equal to the height map at (i, j).        #
        #===============================================================#
        x = i + ti.random(dtype=float)
        y = j + ti.random(dtype=float)
        z = self.height_map[i, j]

        #===============================================================#
        # Now, we march the ray (x, y, z) forward by small steps until  #
        # it either "hits" the height map, exits the horizontal         #
        # boundaries without hitting the height map, or goes above the  #
        # highest value in the height map.                              #
        #===============================================================#
        while True:
            #===========================================================#
            # test if the ray has exited the x and y boundaries         #
            # without colliding, or if the ray's z value is higher than #
            # the global maximum.                                       #
            # if so, set output to 1.0                                  #
            #===========================================================#
            if not(0 < x < w and 0 < y < h) or z > self.max_value:
                result = 1.0
                break

            #===========================================================#
            # the ray is still above the terrain, so march it forward   #
            # by step size l. This will move the ray towards the next   #
            # bounding box.                                             #
            #===========================================================#
            l = self.get_step_size_to_next_bbox(x, y, z, dx, dy, classic=classic)
            if l == 0.0:
                break

            #===========================================================#
            # Propagate ray.                                            #
            #                                                           #
            # we multiply dz by the cell size to account for the        #
            # shallowness of the terrain.                               #
            #                                                           #
            # cell_size = size of each height map cell relative to      #
            # height.                                                   #
            # if cell_size = 1.0, then the horizontal dimensions (x, y) #
            # are in proportion to the vertical dimension (z)           #
            #===========================================================#
            x += l * dx
            y += l * dy
            z += l * dz * self.cell_size

        return result

    @ti.kernel
    def render(self, dx: float, dy: float, dz: float, classic: bool):
        for i, j in self.pixels:
            self.pixels[i, j] = self.trace(i, j, dx, dy, dz, classic)

    def get_shape(self) -> tuple[int]:
        return self.pixels.shape

    def get_image(self):
        return self.pixels

    @staticmethod
    def get_direction(azimuth, altitude):
        azi_rad = azimuth * np.pi / 180
        alt_rad = altitude * np.pi / 180
        dx = np.cos(alt_rad) * np.cos(azi_rad)
        dy = np.cos(alt_rad) * np.sin(azi_rad)
        dz = np.sin(alt_rad)
        return dx, dy, dz
