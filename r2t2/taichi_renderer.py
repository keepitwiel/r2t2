import taichi as ti
import numpy as np


BLACK = ti.Vector([0.0, 0.0, 0.0])
WHITE = ti.Vector([1.0, 1.0, 1.0])

@ti.data_oriented
class TaichiRenderer:
    def __init__(
        self,
        height_map: np.ndarray,
        map_color: np.ndarray = None,
        canvas_shape: tuple[int, int] = (800, 600),
    ):
        self.w_map, self.h_map = height_map.shape
        self.w_canvas, self.h_canvas = canvas_shape
        self.height_map = ti.field(dtype=float, shape=(self.w_map, self.h_map))
        self.height_map.from_numpy(height_map)
        self.max_value = np.max(height_map)
        self.min_value = np.min(height_map)
        self.map_color = ti.Vector.field(n=3, dtype=float, shape=(self.w_map, self.h_map))
        if map_color is not None:
            self.map_color.from_numpy(map_color)
        else:
            self.map_color.fill(WHITE)
        self.live_canvas = ti.Vector.field(n=3, dtype=float, shape=(self.w_canvas, self.h_canvas))
        self.static_map_color = ti.Vector.field(n=3, dtype=float, shape=(self.w_map, self.h_map))
        self.maxmipmap, self.n_levels = self.initialize_maxmipmap()
        self.brightness = 1.0

    def initialize_maxmipmap(self):
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
    def get_propagation_length(self, x, y, z, dx, dy, max_levels):
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
    def get_hierarchical_index(self, x, dx, step):
        return int(x // step) - (1 if x % step == 0.0 and dx < 0 else 0)

    @ti.func
    def length_to_boundary(self, x, x_boundary, dx):
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
        z = self.height_map[int(x), int(y)]
        c = self.map_color[int(x), int(y)]
        while True:
            dt = self.get_propagation_length(x, y, z, dx, dy, max_levels=self.n_levels)
            if dt == 0.0:
                break
            t += dt
            x += dt * dx
            y += dt * dy
            z += dt * dz
            if x <= 0.0 or x >= self.w_map or y <= 0.0 or y >= self.h_map or z >= self.max_value or t >= l_max:
                result = c
                break

        return result

    @ti.func
    def get_map_xy(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        i: int,
        j: int,
        random_xy: bool,
    ):
        """
        Given a bounding box (x, y, w, h), canvas pixel coordinates (i, j) and a random flag,
        calculate a map coordinate (u, v) that is within the bounding box.

        :param x: relative lower horizontal bound of map to be rendererd
        :param y: relative lower vertical bound of map to be rendered
        :param w: relative width of map to be rendered
        :param h: relative height of map to be rendered
        :param i: horizontal pixel coordinate
        :param j: vertical pixel coordinate
        :param random_xy: if True, randomize coordinates within pixel.
        """
        dx = ti.random(float) if random_xy else 0.5
        dy = ti.random(float) if random_xy else 0.5
        u = ((i + dx) * w / self.w_canvas + x) * self.w_map
        v = ((j + dy) * h / self.h_canvas + y) * self.h_map
        return u, v

    @ti.kernel
    def prerender_taichi(
        self,
        azimuth: float,
        altitude: float,
        spp: int,
        sun_radius: float,
        sun_color: ti.math.vec3,
        sky_color: ti.math.vec3,
        l_max: float,
        random_xy: bool,
    ):
        """
        Prerendering function which combines the map color with
        shadow values that can be re-used again and again.

        The idea is to call this function once, and then use
        render_taichi_static for the actual rendering.
        """
        for i, j in self.static_map_color:
            self.static_map_color[i, j] = BLACK
            for _ in range(spp):
                # trace ray to sun
                u = i + ti.random(float) if random_xy else i + 0.5
                v = j + ti.random(float) if random_xy else j + 0.5

                dx, dy, dz = self.get_direction(azimuth, altitude, sun_radius)
                self.static_map_color[i, j] += sun_color * self.collide(
                    u, v, dx, dy, dz, l_max,
                ) / spp / 2

                # trace ray to sky
                u = i + ti.random(float) if random_xy else i + 0.5
                v = j + ti.random(float) if random_xy else j + 0.5

                az = ti.random(float) * 360.0
                al = ti.asin(ti.random(float)) * 90.0
                dx, dy, dz = self.get_direction(az, al, 0.0)
                self.static_map_color[i, j] += sky_color * self.collide(
                    u, v, dx, dy, dz, l_max,
                ) / spp / 2


    @ti.kernel
    def render_taichi_static(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        random_xy: bool,
        brightness: float,
    ):
        """
        Similar to render_taichi_live, except instead of ray tracing to obtain
        shadow values, we lookup these values from a prerendererd image (rendered
        using function prerender_taichi).
        """
        for i, j in self.live_canvas:
            self.live_canvas[i, j] = BLACK
            u, v = self.get_map_xy(x, y, w, h, i, j, random_xy)
            if 0 <= u < self.w_map and 0 <= v < self.h_map:
                self.live_canvas[i, j] = self.static_map_color[int(u), int(v)]
                self.live_canvas[i, j] = (brightness * self.live_canvas[i, j]) ** (1.0 / 2.2)

    @ti.kernel
    def render_taichi_live(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        azimuth: float,
        altitude: float,
        spp: int,
        sun_radius: float,
        sun_color: ti.math.vec3,
        sky_color: ti.math.vec3,
        l_max: float,
        random_xy: bool,
        brightness: float,
    ):
        """
        Main rendering function. This is a very basic
        ray tracing algorithm that either finds a light source
        or not for each collision call. There is no path tracing.

        Per pixel (that is inside bounding box (x, y, w, h)
        we try to collect light from two sources:
        the sun and the sky. We do this by sending
        rays out from each pixel. Because we only look
        at the map top-down, we can start each ray from
        the surface of the map directly.

        For each light source we get a ray direction which
        is then fed into the `collide` function. If the function
        finds a clear line to the source, it returns the map
        color; otherwise black. This map color is then multiplied
        by the light source color.

        :param x: relative lower horizontal bound of map to be rendererd
        :param y: relative lower vertical bound of map to be rendered
        :param w: relative width of map to be rendered
        :param h: relative height of map to be rendered
        :param azimuth: Sun azimuth (degrees)
        :param altitude: Sun altitude above horizon (degrees)
        :param spp: samples per pixel. For spp=1, each light source is sampled once.
        :param sun_radius: radius of Sun in degrees. This determines
            how wide the Sun is sampled.
            0.0 = infinitely distant point source
            5.0 = a square Sun with width and height = 2 * sun_radius
        :param sun_color: color of the Sun
        :param sky_color: color of the Sky
        :param l_max: maximum ray length. a shorter length should reduce
            calculation time at the expense of long shadows
        :param random_xy: boolean flag. If True, the starting point within
            the pixel is sampled randomly. If False, we take the pixel midpoint.
        :param brightness: scalar factor to adjust how dark or how light the map
            appears.
        :return: None
        """

        for i, j in self.live_canvas:
            self.live_canvas[i, j] = BLACK

            # first, trace ray to sun
            u, v = self.get_map_xy(x, y, w, h, i, j, random_xy)
            if 0 <= u < self.w_map and 0 <= v < self.h_map:
                for _ in range(spp):
                    # trace ray to sun
                    dx, dy, dz = self.get_direction(azimuth, altitude, sun_radius)
                    self.live_canvas[i, j] += sun_color * self.collide(
                        u, v, dx, dy, dz, l_max
                    ) / spp / 2

            # then, trace ray to sky
            u, v = self.get_map_xy(x, y, w, h, i, j, random_xy)
            if 0 <= u < self.w_map and 0 <= v < self.h_map:
                for _ in range(spp):
                    # trace ray to sky
                    az = ti.random(float) * 360.0
                    al = ti.asin(ti.random(float)) * 90.0
                    dx, dy, dz = self.get_direction(az, al, 0.0)
                    self.live_canvas[i, j] += sky_color * self.collide(
                        u, v, dx, dy, dz, l_max
                    ) / spp / 2

            # gamma correction
            self.live_canvas[i, j] = (brightness * self.live_canvas[i, j]) ** (1.0 / 2.2)

    def get_image(self):
        return self.live_canvas.to_numpy().astype(np.float32)

    @staticmethod
    @ti.func
    def get_direction(azimuth, altitude, sun_radius):
        u = sun_radius * (ti.random(float) - 0.5)
        v = sun_radius * (ti.random(float) - 0.5)
        azi_rad = (azimuth + u) * np.pi / 180
        alt_rad = (altitude + v) * np.pi / 180
        dx = ti.cos(alt_rad) * ti.cos(azi_rad)
        dy = ti.cos(alt_rad) * ti.sin(azi_rad)
        dz = ti.sin(alt_rad)
        return dx, dy, dz