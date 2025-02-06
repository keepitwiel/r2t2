import taichi as ti
import numpy as np


from .taichi_renderer import TaichiRenderer


class Renderer(TaichiRenderer):
    def __init__(
        self,
        height_map: np.ndarray,
        map_color: np.ndarray = None,
        canvas_shape: tuple[int, int] = (800, 600),
    ):
        """
        Wrapper around TaichiRenderer.

        Because TaichiRenderer is a taichi data-oriented class,
        its attributes cannot be changed throughout its lifetime.

        To provide this functionality, Renderer contains all the
        necessary attributes which are passed to TaichiRenderer.
        """
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

    def render(self, bbox: tuple[int, int, int, int] | None = None, use_static: bool = False):
        """
        Function that calls a render function in the
        TaichiRenderer class.

        Depending on the `use_static` flag, it either calls
        `render_taichi_live` or `render_taichi_static`. The
        former does "live" path tracing, and therefore might
        be slow. The latter uses a static map color that
        has been shaded once by `render_taichi_live` and should
        be faster.
        """
        if bbox is None:
            bbox = [0, 0, self.w_canvas, self.h_canvas]
        if not use_static:
            self.render_taichi_live(
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
                brightness=self.brightness,
            )
        else:
            self.render_taichi_static(
                x=bbox[0],
                y=bbox[1],
                w=bbox[2],
                h=bbox[3],
                zoom=self.zoom,
                x_offset=self.x_offset,
                y_offset=self.y_offset,
                random_xy=self.random_xy,
                brightness=self.brightness,
            )

    def prerender(self):
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
        )
