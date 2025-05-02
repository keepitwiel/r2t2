import taichi as ti
import numpy as np


from .taichi_renderer import TaichiRenderer


class Renderer(TaichiRenderer):
    def __init__(
        self,
        height_map: np.ndarray,
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
        super().__init__(height_map, canvas_shape)
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

    def render(self, map_color: np.ndarray) -> None:
        """
        Function that calls a render function in the
        TaichiRenderer class.

        :return: None
        """
        x, y, w, h = self.get_bbox()

        self.render_taichi(
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
            map_color=map_color,
        )

    def increment_height_map(self, x: int, y: int, dz: float):
        self.height_map[x, y] += dz
        self.initialize_maxmipmap()

    def set_l_max_override(self, x: int, y: int, value: float):
        self.ray_length_override_map[x, y] = value
