import taichi as ti
import numpy as np


@ti.data_oriented
class SimpleReliefMapper:
    def __init__(self, height_map: np.ndarray, map_color: np.ndarray = None, cell_size=1.0):
        self.w, self.h = height_map.shape
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.height_map = ti.field(dtype=float, shape=(self.w, self.h))
        self.height_map.from_numpy(height_map)
        self.max_value = np.max(height_map)
        self.min_value = np.min(height_map)
        self.map_color = ti.Vector.field(n=3, dtype=float, shape=(self.w, self.h))
        if map_color is not None:
            self.map_color.from_numpy(map_color)
        else:
            self.map_color.fill(ti.Vector([1.0, 1.0, 1.0]))
        self.pixels = ti.Vector.field(n=3, dtype=float, shape=(self.w, self.h))
        self.cell_size = cell_size
        self.maxmipmap, self.n_levels = self.initialize_maxmipmap()
        pass

    def initialize_maxmipmap(self):
        # calculate maximum mipmap using the height map as source image.
        #
        # If the source image does not have width/height that are both
        # 2**n (n integer), we force it to be 2**n for convenience purposes.
        #
        # First, calculate the maximum between the log of width and height.
        w, h = self.w, self.h
        log_w = np.log2(w)
        log_h = np.log2(h)
        max_log = max(log_w, log_h)

        # if the log is not an integer, we round up - this is the number of levels
        n_levels = int(max_log) + (1 if max_log != int(max_log) else 0)
        # print(n_levels)

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
    def get_propagation_length_classic(self, x, y, z, dx, dy) -> float:
        result = 0.0
        i = int(x // 1) - (1 if x % 1 == 0.0 and dx < 0 else 0)
        j = int(y // 1) - (1 if y % 1 == 0.0 and dy < 0 else 0)
        z_max = self.height_map[i, j]
        if z >= z_max:
            lx = (1 if x % 1 == 0 else x % 1) / abs(dx) if dx != 0.0 else np.inf
            ly = (1 if y % 1 == 0 else y % 1) / abs(dy) if dy != 0.0 else np.inf
            result = min(lx, ly)
        return result


    @ti.func
    def get_propagation_length(self, x, y, z, dx, dy, max_levels):
        result = 0.0
        step = 0.0
        offset = 0
        for level in range(max_levels + 1):
            step = 2**level
            i = int(x // step) - (1 if x % step == 0.0 and dx < 0 else 0)  # the bug should be in here somewhere
            j = int(y // step) - (1 if y % step == 0.0 and dy < 0 else 0)  # the bug should be in here somewhere
            z_max = self.height_map[i, j] if level == 0 else self.maxmipmap[i, offset + j]
            if z < z_max:
                step = step // 2
                break
            k = self.n_levels - level
            offset += 2**k if level > 0 else 0

        if step >= 1:
            i = int(x // step) - (1 if x % step == 0.0 and dx < 0 else 0)
            j = int(y // step) - (1 if y % step == 0.0 and dy < 0 else 0)
            # print(i * step, x, (i + 1) * step)
            # print(j * step, y, (j + 1) * step)
            l_left = self.length_to_boundary(x, i * step, dx)
            l_right = self.length_to_boundary(x, (i + 1) * step, dx)
            l_top = self.length_to_boundary(y, j * step, dy)
            l_bottom = self.length_to_boundary(y, (j + 1) * step, dy)
            result = min(l_left, l_right, l_top, l_bottom)

            # lx = (step if x % step == 0.0 else x % step) / abs(dx) if dx != 0.0 else np.inf
            # ly = (step if y % step == 0.0 else y % step) / abs(dy) if dy != 0.0 else np.inf
            # result = min(lx, ly)

        return result

    @ti.func
    def length_to_boundary(self, x, x_boundary, dx):
        result = (x_boundary - x) / dx if dx != 0 else np.inf
        result = result if result > 0 else np.inf
        return result

    @ti.kernel
    def maxmipmap_debug(
            self, i_: int, j_: int, zoom: float, azimuth: float, altitude: float, n_levels: int
    ) -> tuple[int, float, float, float]:
        l = 0.0
        x_ = i_ + 0.5
        y_ = j_ + 0.5
        x = self.x_offset + x_ / zoom
        y = self.y_offset + y_ / zoom
        z = self.height_map[int(x), int(y)]
        h = z
        dx, dy, dz = self.get_direction(azimuth, altitude, sun_width=0.0)
        counter = 0
        dl = 0.0
        i, j = 0, 0
        while True:
            counter += 1
            dl = 0.0

            # first, get step size
            step = 0.0
            offset = 0
            for level in range(n_levels + 1):
                step = 2 ** level
                i = int(x // step) - (1 if x % step == 0.0 and dx < 0 else 0)
                j = int(y // step) - (1 if y % step == 0.0 and dy < 0 else 0)
                z_max = self.height_map[i, j] if level == 0 else self.maxmipmap[i, offset + j]
                if z < z_max:
                    step = step // 2
                    break
                k = self.n_levels - level
                offset += 2 ** k if level > 0 else 0

            # using step size, determine propagation length
            if step >= 1:
                i = int(x // step) - (1 if x % step == 0.0 and dx < 0 else 0)
                j = int(y // step) - (1 if y % step == 0.0 and dy < 0 else 0)
                # print(i * step, x, (i + 1) * step)
                # print(j * step, y, (j + 1) * step)
                l_left = self.length_to_boundary(x, i * step, dx)
                l_right = self.length_to_boundary(x, (i + 1) * step, dx)
                l_top = self.length_to_boundary(y, j * step, dy)
                l_bottom = self.length_to_boundary(y, (j + 1) * step, dy)
                dl = min(l_left, l_right, l_top, l_bottom)

                # #####
                # lx = (step if x % step == 0.0 else x % step) / abs(dx) if dx != 0.0 else np.inf  # 19.966
                # ly = (step if y % step == 0.0 else y % step) / abs(dy) if dy != 0.0 else np.inf  # 26.143
                # dl = min(lx, ly)

            if i_ == 233 and j_ == 263:
                print(
                    f"i={i_:03d}, j={j_:03d}, x={x:03.3f}, y={y:03.3f}, z={z:03.3f}, "
                    f"dx={dx:03.3f}, dy={dy:03.3f}, dz={dz:03.3f}, counter={counter:03d}, "
                    f"step={step:03.3f}, dl={dl:03.3f}"
                )

            if dl == 0.0 or z + dl * dz * self.cell_size > self.max_value:
                break

            l += dl
            x += dl * dx
            y += dl * dy
            z += dl * dz * self.cell_size
            if x <= 0.0 or x >= self.w or y <= 0.0 or y >= self.h or counter > 1000:
                break
        return counter, l, h, dl

    @ti.func
    def collide(
        self,
        i: int,
        j: int,
        dx: float,
        dy: float,
        dz: float,
        maxmipmap: bool,
        zoom: float,
        l_max: float,
        random_xy: bool,
    ):
        l = 0.0
        result = ti.Vector([0.0, 0.0, 0.0])
        dx_ = ti.random(dtype=float) if random_xy else 0.5
        dy_ = ti.random(dtype=float) if random_xy else 0.5
        x_ = i + dx_
        y_ = j + dy_
        x = self.x_offset + x_ / zoom
        y = self.y_offset + y_ / zoom
        if 0 <= x < self.w and 0 <= y < self.h:
            z = self.height_map[int(x), int(y)]
            c = self.map_color[int(x), int(y)]
            if not maxmipmap:
                while True:
                    if z >= self.max_value:
                        result = c
                        break
                    else:
                        dl = self.get_propagation_length(x, y, z, dx, dy, max_levels=0)
                        if dl == 0.0:
                            break
                        l += dl
                        x += dl * dx
                        y += dl * dy
                        z += dl * dz * self.cell_size

                        if x <= 0.0 or x >= self.w or y <= 0.0 or y >= self.h or l >= l_max:
                            result = c
                            break
            else:
                while True:
                    dl = self.get_propagation_length(x, y, z, dx, dy, max_levels=self.n_levels)
                    if dl == 0.0:
                        break

                    l += dl
                    x += dl * dx
                    y += dl * dy
                    z += dl * dz * self.cell_size

                    if x <= 0.0 or x >= self.w or y <= 0.0 or y >= self.h or l >= l_max:
                        result = c
                        break

        return result

    @ti.kernel
    def render(
            self,
            azimuth: float,
            altitude: float,
            maxmipmap: bool,
            zoom: float,
            spp: int,
            sun_width: float,
            sun_color: ti.math.vec3,
            sky_color: ti.math.vec3,
            l_max: float,
            random_xy: bool,
    ):
        for i, j in self.pixels:
            self.pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])
            for _ in range(spp):
                # trace path to sun
                dx, dy, dz = self.get_direction(azimuth, altitude, sun_width)
                self.pixels[i, j] += sun_color * self.collide(
                    i, j, dx, dy, dz, maxmipmap, zoom, l_max, random_xy
                ) / spp / 2

                # trace path to sky
                az = ti.random(float) * 360.0
                al = ti.asin(ti.random(float)) * 90.0
                dx, dy, dz = self.get_direction(az, al, 0.0)
                self.pixels[i, j] += sky_color * self.collide(
                    i, j, dx, dy, dz, maxmipmap, zoom, l_max, random_xy
                ) / spp / 2

            # gamma correction
            self.pixels[i, j] = self.pixels[i, j] ** (1.0/2.2)

    def get_shape(self) -> tuple[int]:
        return self.pixels.shape

    def get_image(self):
        return self.pixels.to_numpy().astype(np.float32)

    @staticmethod
    @ti.func
    def get_direction(azimuth, altitude, sun_width):
        u = sun_width * (ti.random(float) - 0.5)
        v = sun_width * (ti.random(float) - 0.5)
        azi_rad = (azimuth + u) * np.pi / 180
        alt_rad = (altitude + v) * np.pi / 180
        dx = ti.cos(alt_rad) * ti.cos(azi_rad)
        dy = ti.cos(alt_rad) * ti.sin(azi_rad)
        dz = ti.sin(alt_rad)
        return dx, dy, dz
