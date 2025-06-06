import numpy as np


class M4:
    def __init__(self, dimension: int, mx: np.ndarray, my: np.ndarray, z: np.ndarray):
        log2n = np.log2(dimension)
        self.n_levels = int(log2n)
        assert log2n == self.n_levels, "dimension has to be exactly a power of 2."
        self.dimension = dimension
        self.mx = mx.astype(np.float32)
        self.my = my.astype(np.float32)
        self.z = z.astype(np.float32)
        self.minmipmap = self.get_mipmap(is_max=False)
        self.maxmipmap = self.get_mipmap(is_max=True)

    @staticmethod
    def reduce(array: np.ndarray, step_size: int, is_max: bool) -> np.ndarray:
        func = np.max if is_max else np.min
        result = func(
            np.stack(
                [
                    array[  :-1:step_size,  :-1:step_size],
                    array[ 1:  :step_size,  :-1:step_size],
                    array[  :-1:step_size, 1:  :step_size],
                    array[ 1:  :step_size, 1:  :step_size],
                ],
                axis=2
            ),
            axis=2
        )
        return result

    def get_mipmap(self, is_max: bool) -> list[np.ndarray]:
        buffer = self.reduce(self.z, step_size=1, is_max=is_max)
        result = [buffer]
        for level in range(self.n_levels):
            buffer = self.reduce(buffer, step_size=2, is_max=is_max)
            result.append(buffer)
        return result

    def get_array(self, level: int, is_max: bool) -> np.ndarray:
        assert 0 <= level < self.n_levels, "invalid level."
        return self.maxmipmap[level] if is_max else self.minmipmap[level]

    def get_dimension(self) -> int:
        return self.dimension

    def get_n_levels(self) -> int:
        return self.n_levels

    def get_global_max(self):
        return self.maxmipmap[self.n_levels - 1][0, 0]
