import taichi as ti
import numpy as np


from .taichi_renderer import TaichiRenderer


UPDATE_RADIUS = 10


class Renderer(TaichiRenderer):
    def __init__(
        self,
        height_map: np.ndarray,
        map_color: np.ndarray = None,
        canvas_shape: tuple[int, int] = (800, 600),
        **kwargs,
    ):
        """
        Wrapper around TaichiRenderer.

        Because TaichiRenderer is a taichi data-oriented class,
        its attributes cannot be changed throughout its lifetime.

        To provide this functionality, Renderer contains all the
        necessary attributes which are passed to TaichiRenderer.
        """
        super().__init__(height_map, map_color, canvas_shape)
        self.azimuth: float = kwargs.get("azimuth", 45.0)
        self.altitude: float = kwargs.get("altitude", 45.0)
        self.zoom: float = kwargs.get("zoom", 1.0)
        self.x_center: float = kwargs.get("x_center", 0.5)
        self.y_center: float = kwargs.get("y_center", 0.5)
        self.spp: int = kwargs.get("spp", 1)
        self.sun_radius: float = kwargs.get("sun_radius", 2.5)
        self.sun_color: tuple[float, float, float] = kwargs.get("sun_color", (1.0, 0.9, 0.0))
        self.sky_color: tuple[float, float, float] = kwargs.get("sky_color", (0.2, 0.2, 1.0))
        self.l_max: float = 2**self.n_levels
        self.random_xy: bool = kwargs.get("random_xy", True)
        self.static: bool = kwargs.get("static", False)
        
        if self.static:
            self.prerender()

    def get_bbox(self) -> tuple[float, float, float, float]:
        """Returns a relative bounding box consisting of
        lower left coordinates and width/height.

        If self.w_canvas = self.w_map, self.h_canvas = self.h_map
        and self.zoom = 1.0, then the bbox is [0, 0, 1, 1]
        :returns: x, y, w, h.
        """
        w = self.w_canvas / self.w_map / self.zoom
        h = self.h_canvas / self.h_map / self.zoom
        x = self.x_center - 0.5 * w
        y = self.y_center - 0.5 * h
        return x, y, w, h

    def render(self) -> None:
        """
        Function that calls a render function in the
        TaichiRenderer class.

        If self.static is True, this function uses a static map color that
        has been shaded once by `render_taichi_live` and should
        be faster. Otherwise, it does "live" path tracing, which might
        be slower.

        :return: None
        """
        x, y, w, h = self.get_bbox()

        if not self.static:
            self.render_taichi_live(
                x=x,
                y=y,
                w=w,
                h=h,
                azimuth=self.azimuth,
                altitude=self.altitude,
                spp=self.spp,
                sun_radius=self.sun_radius,
                sun_color=self.sun_color,
                sky_color=self.sky_color,
                l_max=self.l_max,
                random_xy=self.random_xy,
                brightness=self.brightness,
            )
        else:
            self.render_taichi_static(
                x=x,
                y=y,
                w=w,
                h=h,
                random_xy=self.random_xy,
                brightness=self.brightness,
            )

    def prerender(self, x: int = 0, y: int = 0, w: int = -1, h: int = -1):
        """
        Prerenders shadows to improve rendering speed when
        calling `render_taichi` with `use_static` flag
        set to True
        """
        self.prerender_taichi(
            azimuth=self.azimuth,
            altitude=self.altitude,
            spp=self.spp,
            sun_radius=self.sun_radius,
            sun_color=self.sun_color,
            sky_color=self.sky_color,
            l_max=self.l_max,
            random_xy=self.random_xy,
            x=x,
            y=y,
            w=w,
            h=h,
        )

    @ti.kernel
    def set_map_color(self, x: int, y: int, color: ti.math.vec3):
        self.map_color[x, y] = ti.math.vec3(color)

    def increment_height_map(self, x: int, y: int, dz: float):
        self.height_map[x, y] += dz
        self.update_maxmipmap(
            x=x - UPDATE_RADIUS, 
            y=y - UPDATE_RADIUS, 
            w=2 * UPDATE_RADIUS + 1, 
            h=2 * UPDATE_RADIUS + 1
        )
