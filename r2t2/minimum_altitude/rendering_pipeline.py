import time

import taichi as ti
import numpy as np


from .fast_mipmap import fast_mipmap
from .quad_altitude import quad_altitude_sampling, simple_grid_sampling


class ShaderPipeline:
    def __init__(self, n_cells: int, cell_size: float):
        n_levels = np.log2(n_cells)
        self.n_levels = int(n_levels)
        assert self.n_levels == n_levels, "n_cells has to be of the form 2^N, where N integer >= 0"
        self.n_cells = n_cells
        self.n_nodes = n_cells + 1
        self.cell_size = cell_size
        self.minmipmap = np.zeros((n_cells * 2 - 1, n_cells), dtype=np.float32)
        self.maxmipmap = np.zeros((n_cells * 2 - 1, n_cells), dtype=np.float32)
    def render(
        self,
        height_field: np.ndarray,
        azimuth: float,
        global_min_altitude: float,
        global_max_altitude: float,
        n_samples: int,
        illumination_field: np.ndarray,
        simple: bool = True,
    ):
        """
        Given a height field of (self.n_nodes x self.n_nodes), apply the rendering
        pipeline that results in an illumination map:
        - calculate mipmaps
        - calculate lower bound on altitude fields (up to a certain level)
        - calculate illumination (get minimum altitude for individual rays and integrate)
        """
        assert height_field.shape == (self.n_nodes, self.n_nodes)
        illumination_field.fill(global_min_altitude)
        fast_mipmap(height_field, out_min=self.minmipmap, out_max=self.maxmipmap)

        if simple:
            simple_grid_sampling(
                height_field, azimuth, global_min_altitude, global_max_altitude, illumination_field
            )
        else:
            buffer = np.zeros(n_samples, dtype=np.float32)
            quad_altitude_sampling(
                height_field=height_field,
                maxmipmap=self.maxmipmap,
                azimuth=azimuth,
                global_min_altitude=global_min_altitude,
                global_max_altitude=global_max_altitude,
                n_levels=self.n_levels,
                n_samples=n_samples,
                buffer=buffer,
                out_array=illumination_field,
            )
