import taichi as ti; ti.init(arch=ti.cpu)
from max_mip_map import MaxMipMap

import numpy as np


EPSILON = 1e-6
NEAR_INFINITE = 1e10


@ti.data_oriented
class SimpleReliefMapper:
    def __init__(self, height_map: np.ndarray, cell_size=10.0):
        w, h = height_map.shape
        self.altitude = np.pi / 2
        self.height_map = ti.field(dtype=float, shape=(w, h))
        self.height_map.from_numpy(height_map)
        self.max_value = np.max(self.height_map.to_numpy())
        self.pixels = ti.field(dtype=float, shape=(w, h))
        self.cell_size = cell_size
        self.maximum_ray_length = NEAR_INFINITE
        self.maxmipmap = MaxMipMap(self.height_map)

    @ti.func
    def get_partial_step_size(self, d: float, x: float) -> float:
        result = 0.0
        if d < 0.0:
            lb = int(x)
            if x % 1.0 == 0.0:
                lb -= 1.0
            result = (x - lb) / -d
        elif d > 0.0:
            ub = int(x) + 1
            result = (ub - x) / d
        elif d == 0.0:
            result = self.maximum_ray_length
        assert result > 0
        return result

    @ti.func
    def max_test(self, x, y, z):
        i, j = x // 1, y // 1
        for k in range(len(self.maxmipmap.levels)):
            mmm = self.maxmipmap.get_value(i, j, level=k)
            z_max = mmm[i, j]
            if z_max < z:
                break
            i, j = i // 2, j // 2
        return z_max, k

    @ti.func
    def get_step_size_to_next_bbox(
            self, x: float, y: float, z: float, dx: float, dy: float
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
        :return:
        """
        z_max, k = self.max_test(x, y, z)
        if z_max > z and k == 0:
            #===========================================================#
            # even at the smallest resolution,                          #
            # the maximum height is above z.                            #
            # therefore we conclude the ray is stopped.                 #
            #===========================================================#
            return 0.0
        lx = self.get_partial_step_size(dx, x)
        ly = self.get_partial_step_size(dy, y)
        l = min(lx, ly)
        l += EPSILON
        return l

    @ti.func
    def trace(self, i, j, dx, dy, dz, amplitude):
        result = 0.0
        w, h = self.height_map.shape

        #===============================================================#
        # within cell (i, j), randomly pick an (x, y) coordinate.       #
        # the z coordinate is equal to the height map at (i, j).        #
        #===============================================================#
        x = i + ti.random(dtype=float)
        y = j + ti.random(dtype=float)
        z = self.height_map[i, j] * amplitude

        #===============================================================#
        # Now, we march the ray (x, y, z) forward by small steps until  #
        # it either "hits" the height map, exits the horizontal         #
        # boundaries without hitting the height map, or goes above the  #
        # highest value in the height map.                              #
        #===============================================================#
        while True:
            #===========================================================#
            # test if the ray has exited the x and y boundaries         #
            # without colliding.                                        #
            # if so, set output to 1.0                                  #
            #===========================================================#
            if not(0 < x < w and 0 < y < h):
                result = 1.0
                break

            #===========================================================#
            # ray is still within x and y boundaries -                  #
            # test if the height map at the (x, y) coordinate           #
            # is above the ray's z coordinate.                          #
            # if so, the ray has hit the height map, and we return 0.0  #
            #===========================================================#
            i_ = int(x)
            j_ = int(y)
            z_ = self.height_map[i_, j_] * amplitude
            if z_ > z:
                break

            #===========================================================#
            # the ray is still above the terrain, so march it forward   #
            # by step size l. This will move the ray towards the next   #
            # bounding box.                                             #
            #===========================================================#
            l = self.get_step_size_to_next_bbox(x, y, z, dx, dy)
            if l == 0.0:
                break

            #===========================================================#
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

            # #===========================================================#
            # # finally, if the ray's z value is higher than the entire   #
            # # height map, it means it will never collide with it        #
            # # (assuming dz > 0).                                        #
            # #                                                           #
            # # this is a safe but costly approach which could be         #
            # # improved by checking nearby maximum values instead of the #
            # # global maximum.                                           #
            # #===========================================================#
            # if z > max_value:
            #     result = 1.0
            #     break

        return result

    @ti.kernel
    def render(self, dx: float, dy: float, dz: float, amplitude: float):
        for i, j in self.pixels:
            self.pixels[i, j] = self.trace(i, j, dx, dy, dz, amplitude)

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
