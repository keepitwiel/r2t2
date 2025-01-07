import taichi as ti; ti.init(arch=ti.cpu)
import numpy as np
from noise import snoise2


EPSILON = 1e-6
NEAR_INFINITE = 1e10


def simplex_height_map(dim, octaves, amplitude, seed=42):
    """
    Simple wrapper that samples a height map from simplex noise.
    :param dim:
    :param octaves:
    :param amplitude:
    :param seed:
    :return:
    """
    arr = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            arr[j, i] = snoise2(j / dim, i / dim, octaves=octaves, base=seed)
    arr *= amplitude
    return arr


@ti.data_oriented
class SimpleReliefMapper:
    def __init__(self, height_map: np.ndarray, cell_size=10.0):
        w, h = height_map.shape
        self.max_t = max(w, h)
        self.altitude = np.pi / 2
        self.height_map = ti.field(dtype=float, shape=(w, h))
        self.height_map.from_numpy(height_map)
        self.max_value = np.max(self.height_map.to_numpy())
        self.pixels = ti.field(dtype=float, shape=(w, h))
        self.cell_size = cell_size
        self.maximum_ray_length = NEAR_INFINITE

    @ti.func
    def get_partial_length(self, d: float, x: float) -> float:
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
    def get_ray_length(self, x: float, y: float, dx: float, dy: float) -> float:
        lx = self.get_partial_length(dx, x)
        ly = self.get_partial_length(dy, y)
        l = min(lx, ly)
        l += EPSILON
        return l

    @ti.func
    def trace(self, i, j, dx, dy, dz, amplitude, max_value):
        result = 0.0
        w, h = self.height_map.shape

        #===============================================================#
        # within cell (i, j), randomly pick an (x, y) coordinate.       #
        # the z coordinate is equal to the height map at (i, j).        #
        x = i + ti.random(dtype=float)                                  #
        y = j + ti.random(dtype=float)                                  #
        z = self.height_map[i, j] * amplitude                           #
        #===============================================================#

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
            # by step size l.                                           #
            # we multiply dz by the cell size to account for the        #
            # shallowness of the terrain.                               #
            #                                                           #
            # cell_size = size of each height map cell relative to      #
            # height.                                                   #
            # if cell_size = 1.0, then the horizontal dimensions (x, y) #
            # are in proportion to the vertical dimension (z)           #
            #===========================================================#
            l = self.get_ray_length(x, y, dx, dy)
            # if l < 2 * EPSILON:
            #    print(f"l: {l:.7f}, x, y, dx, dy: {x:.3f}, {y:.3f}, {dx:.3f}, {dy:.3f}")
            x += l * dx
            y += l * dy
            z += l * dz * self.cell_size

            #===========================================================#
            # finally, if the ray's z value is higher than the entire   #
            # height map, it means it will never collide with it        #
            # (assuming dz > 0).                                        #
            #                                                           #
            # this is a safe but costly approach which could be         #
            # improved by checking nearby maximum values instead of the #
            # global maximum.                                           #
            #===========================================================#
            if z > max_value:
                result = 1.0
                break

        return result

    @ti.kernel
    def render(self, dx: float, dy: float, dz: float, amplitude: float):
        max_value = self.max_value * amplitude
        for i, j in self.pixels:
            self.pixels[i, j] = self.trace(i, j, dx, dy, dz, amplitude, max_value)

    def get_shape(self) -> tuple[int]:
        return self.pixels.shape

    def get_image(self):
        return self.pixels


def get_direction(azimuth, altitude):
    azi_rad = azimuth * np.pi / 180
    alt_rad = altitude * np.pi / 180
    dx = np.cos(alt_rad) * np.cos(azi_rad)
    dy = np.cos(alt_rad) * np.sin(azi_rad)
    dz = np.sin(alt_rad)
    return dx, dy, dz


def run(renderer):
    window = ti.ui.Window(name='Window Title', res=renderer.get_shape(), fps_limit=30, pos=(0, 0))
    gui = window.get_gui()
    canvas = window.get_canvas()
    vertical_scale = 1.0
    azimuth = 45 # light source horizontal direction, degrees
    altitude = 15 # degrees
    while window.running:
        with gui.sub_window("Sub Window", 0.1, 0.1, 0.5, 0.2):
            vertical_scale = gui.slider_float("vertical scale", vertical_scale, 0.0, 10.0)
            altitude = gui.slider_float("altitude (deg)", altitude, 0, 89)
        dx, dy, dz = get_direction(azimuth, altitude)

        renderer.render(dx, dy, dz, vertical_scale)
        canvas.set_image(renderer.get_image())
        window.show()

        # the following rotates the azimuth between 0 and 360 degrees, with increments of 1 degree per step
        azimuth = (azimuth + 1) % 360


def example_map_1(n):
    octaves = int(np.log2(n))
    z = simplex_height_map(dim=n, octaves=octaves, amplitude=n, seed=42)
    z = np.float32(z)
    z[n // 2 - n // 8:n // 2 + n // 8, n // 2 - n // 8:n // 2 + n // 8] = 0
    return z


def main(n):
    # define height map
    z = example_map_1(n)

    # initialize renderer
    renderer = SimpleReliefMapper(height_map=z)

    # run app
    run(renderer)


if __name__ == "__main__":
    main(n=512)
