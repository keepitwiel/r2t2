import taichi as ti
import numpy as np


@ti.data_oriented
class TaichiRenderer:
    def __init__(
        self,
        height_map: np.ndarray,
        canvas_shape: tuple[int, int] = (800, 600),
    ):
        self.w_map, self.h_map = height_map.shape
        self.w_canvas, self.h_canvas = canvas_shape
        self.height_map = ti.field(dtype=float, shape=(self.w_map, self.h_map))
        self.height_map.from_numpy(height_map)
        self.max_value = np.max(height_map)
        self.min_value = np.min(height_map)
        self.map_color = ti.Vector.field(n=3, dtype=float, shape=(self.w_map, self.h_map))
        self.live_canvas = ti.Vector.field(n=3, dtype=float, shape=(self.w_canvas, self.h_canvas))
        self.ray_length_override_map = ti.field(dtype=float, shape=(self.w_map, self.h_map))
        self.ray_length_override_map.fill(np.inf)
        self.maxmipmap, self.n_levels = self.initialize_maxmipmap()
        self.brightness = 1.0
        self.prerendered = False

    def initialize_maxmipmap(self) -> tuple[np.ndarray, int]:
        """
        # calculate maximum mipmap using the height map as source image.
        #
        # If the source image does not have width/height that are both
        # 2**n (n integer), we force it to be 2**n for convenience purposes.
        """

        # First, calculate the maximum between the log of width and height.
        w, h = self.w_map, self.h_map
        log_w = np.log2(w)
        log_h = np.log2(h)
        max_log = max(log_w, log_h)

        # if the log is not an integer, we round up - this is the number of levels
        n_levels = int(max_log) + (1 if max_log != int(max_log) else 0)

        # Then, calculate the dimension parameter of the mipmap.
        dim = int(2**n_levels)

        # we create a mipmap field using the dimension parameter
        result = ti.field(
            shape=(dim // 2, dim - 1),
            dtype=float
        )
        result.fill(-np.inf)

        maxmipmap = result.to_numpy()
        z = -np.inf + np.zeros((dim, dim))
        z[:w, :h] = self.height_map.to_numpy()[:w, :h]

        y_offset = 0
        for level in range(n_levels):
            dim = dim // 2
            z_ = np.stack([z[::2, ::2], z[::2, 1::2], z[1::2, ::2], z[1::2, 1::2]], axis=2)
            z = np.max(z_, axis=2)
            maxmipmap[0:dim, y_offset:y_offset + dim] = z
            y_offset += dim

        result.from_numpy(maxmipmap)
        return result, n_levels


    @ti.func
    def get_propagation_length(self, x: float, y: float, z: float, dx: float, dy: float, max_levels: int):
        """
        Given
        - a position (x, y, z) in ray traversion space and
        - a direction (dx, dy) in map space,

        - determine a bounding box around (x, y) depending on
          which maxmipmap level we are (determined by z);
        - find a point (x', y') = (x + l * dx, y + l * dy)
          on the bounding box that is closest to (x, y);
        - return l.

        This is the guaranteed length a ray can travel without
        colliding with the height map.
        """
        result = 0.0
        step = 0.0
        offset = 0

        for level in range(max_levels + 1):
            step = 2**level
            i = self.get_hierarchical_index(x, dx, step)
            j = self.get_hierarchical_index(y, dy, step)
            z_max = self.height_map[i, j] if level == 0 else self.maxmipmap[i, offset + j]
            if z < z_max:
                step = step // 2
                break
            k = self.n_levels - level
            offset += 2**k if level > 0 else 0

        if step >= 1:
            i = self.get_hierarchical_index(x, dx, step)
            j = self.get_hierarchical_index(y, dy, step)
            l_left = self.length_to_boundary(x, i * step, dx)
            l_right = self.length_to_boundary(x, (i + 1) * step, dx)
            l_top = self.length_to_boundary(y, j * step, dy)
            l_bottom = self.length_to_boundary(y, (j + 1) * step, dy)
            result = min(l_left, l_right, l_top, l_bottom)

        return result

    @ti.func
    def get_hierarchical_index(self, x: float, dx: float, step: float):
        return int(x // step) - (1 if x % step == 0.0 and dx < 0 else 0)

    @ti.func
    def length_to_boundary(self, x: float, x_boundary: float, dx: float):
        result = np.inf
        if dx != 0:
            result = (x_boundary - x) / dx
            result = result if result > 0 else np.inf
        return result

    @ti.func
    def collide(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        dz: float,
        l_max: float,
    ):
        t = 0.0
        result = BLACK
        i = int(x)
        j = int(y)
        z = self.height_map[i, j]

        while True:
            dt = self.get_propagation_length(x, y, z, dx, dy, max_levels=self.n_levels)
            if dt == 0.0:
                break
            t += dt
            x += dt * dx
            y += dt * dy
            z += dt * dz
            if x <= 0.0 or x >= self.w_map or y <= 0.0 or y >= self.h_map or z >= self.max_value or t >= l_max_override:
                result = WHITE
                break

        return result
