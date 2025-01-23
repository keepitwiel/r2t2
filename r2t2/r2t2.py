import taichi as ti
import numpy as np


BLACK = ti.Vector([0.0, 0.0, 0.0])


@ti.data_oriented
class BaseRenderer:
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
            self.map_color.fill(ti.Vector([1.0, 1.0, 1.0]))
        self.pixels = ti.Vector.field(n=3, dtype=float, shape=(self.w_canvas, self.h_canvas))
        self.maxmipmap, self.n_levels = self.initialize_maxmipmap()


    def initialize_maxmipmap(self):
        # calculate maximum mipmap using the height map as source image.
        #
        # If the source image does not have width/height that are both
        # 2**n (n integer), we force it to be 2**n for convenience purposes.
        #
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
        i: int,
        j: int,
        x_offset: float,
        y_offset: float,
        dx: float,
        dy: float,
        dz: float,
        zoom: float,
        l_max: float,
        random_xy: bool,
    ):
        t = 0.0
        result = ti.Vector([0.0, 0.0, 0.0])
        dx_ = ti.random(dtype=float) if random_xy else 0.5
        dy_ = ti.random(dtype=float) if random_xy else 0.5
        x_ = i + dx_
        y_ = j + dy_
        x = x_offset + x_ / zoom
        y = y_offset + y_ / zoom
        if 0 <= x < self.w_map and 0 <= y < self.h_map:
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

    @ti.kernel
    def render_internal(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        azimuth: float,
        altitude: float,
        zoom: float,
        x_offset: float,
        y_offset: float,
        spp: int,
        sun_radius: float,
        sun_color: ti.math.vec3,
        sky_color: ti.math.vec3,
        l_max: float,
        random_xy: bool,
    ):
        """
        Main rendering function. This is a very basic
        ray tracing algorithm that either finds a light source
        or not for each pixel. There is no path tracing.

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

        :param x: lower bound of horizontal pixel range to be rendererd
        :param y: lower bound of vertical pixel range to be rendered
        :param w: width of pixel range to be rendered
        :param h: height of pixel range to be rendered
        :param azimuth: Sun azimuth (degrees)
        :param altitude: Sun altitude above horizon (degrees)
        :param zoom: zoom factor. 1.0 = normal.
        :param x_offset: horizontal camera offset
        :param y_offset: vertical camera offset
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
        :return: None
        """
        for i, j in self.pixels:
            if x <= i < x + w and y <= j < y + w:
                self.pixels[i, j] = BLACK
                for _ in range(spp):
                    # trace ray to sun. TODO: get x, y first, then determine if its on map, then collide
                    dx, dy, dz = self.get_direction(azimuth, altitude, sun_radius)
                    self.pixels[i, j] += sun_color * self.collide(
                        i, j, x_offset, y_offset, dx, dy, dz, zoom, l_max, random_xy
                    ) / spp / 2

                    # trace ray to sky. TODO: get x, y first, then determine if its on map, then collide
                    az = ti.random(float) * 360.0
                    al = ti.asin(ti.random(float)) * 90.0
                    dx, dy, dz = self.get_direction(az, al, 0.0)
                    self.pixels[i, j] += sky_color * self.collide(
                        i, j, x_offset, y_offset, dx, dy, dz, zoom, l_max, random_xy
                    ) / spp / 2

                # gamma correction
                self.pixels[i, j] = self.pixels[i, j] ** (1.0/2.2)

    @ti.kernel
    def render_bbox(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        azimuth: float,
        altitude: float,
        zoom: float,
        x_offset: float,
        y_offset: float,
        spp: int,
        sun_radius: float,
        sun_color: ti.math.vec3,
        sky_color: ti.math.vec3,
        l_max: float,
        random_xy: bool,
    ):
        """
        Same as the render_internal function, except the rendering is limited to
        pixels within
        """
        for i, j in self.pixels:
            self.pixels[i, j] = BLACK
            for _ in range(spp):
                # trace ray to sun. TODO: get x, y first, then determine if its on map, then collide
                dx, dy, dz = self.get_direction(azimuth, altitude, sun_radius)
                self.pixels[i, j] += sun_color * self.collide(
                    i, j, x_offset, y_offset, dx, dy, dz, zoom, l_max, random_xy
                ) / spp / 2

                # trace ray to sky. TODO: get x, y first, then determine if its on map, then collide
                az = ti.random(float) * 360.0
                al = ti.asin(ti.random(float)) * 90.0
                dx, dy, dz = self.get_direction(az, al, 0.0)
                self.pixels[i, j] += sky_color * self.collide(
                    i, j, x_offset, y_offset, dx, dy, dz, zoom, l_max, random_xy
                ) / spp / 2

            # gamma correction
            self.pixels[i, j] = self.pixels[i, j] ** (1.0/2.2)


    def get_image(self):
        return self.pixels.to_numpy().astype(np.float32)

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


class Renderer(BaseRenderer):
    def __init__(
        self,
        height_map: np.ndarray,
        map_color: np.ndarray = None,
        canvas_shape: tuple[int, int] = (800, 600),
    ):
        super().__init__(height_map, map_color, canvas_shape)
        self.azimuth: float = 45.0
        self.altitude: float = 45.0
        self.zoom: float = 1.0
        self.x_offset: float = 0.0
        self.y_offset: float = 0.0
        self.spp: int = 1
        self.sun_radius: float = 2.5
        self.sun_color: tuple[float, float, float] = (1.0, 0.9, 0.0)
        self.sky_color: tuple[float, float, float] = (0.2, 0.2, 1.0)
        self.l_max: float = 2**self.n_levels
        self.random_xy: bool = True

    def render(self, bbox: tuple[int, int, int, int] | None = None):
        if bbox is None:
            bbox = [0, 0, self.w_canvas, self.h_canvas]
        self.render_internal(
            x=bbox[0],
            y=bbox[1],
            w=bbox[2],
            h=bbox[3],
            azimuth=self.azimuth,
            altitude=self.altitude,
            zoom=self.zoom,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            spp=self.spp,
            sun_radius=self.sun_radius,
            sun_color=self.sun_color,
            sky_color=self.sky_color,
            l_max=self.l_max,
            random_xy=self.random_xy,
        )
