import numpy as np


class MaxMipMap:
    def __init__(self, z):
        self.z = z
        self.w, self.h = z.shape
        self.dtype = z.dtype
        self.maxmipmap, self.levels = self.get_maxmipmap()

    def get_maxmipmap(self):
        w, h = self.w, self.h
        z = self.z

        result = np.zeros(
            shape=(w // 2 + 1 if w % 2 != 0 else w // 2, h),
            dtype=self.dtype
        )
        y0 = 0

        level = 1
        while min(z.shape) > 1:
            w, h = z.shape
            w_ = w // 2 + (1 if w % 2 != 0 else 0)  # Handle odd width
            h_ = h // 2 + (1 if h % 2 != 0 else 0)  # Handle odd height
            print(f"Level: {level}, w x h: {w_} x {h_}")
            mipmap = np.zeros(shape=(w_, h_), dtype=self.dtype)

            for i in range(w_):
                for j in range(h_):
                    # Define the window for max calculation
                    i_end = min((i + 1) * 2, w)
                    j_end = min((j + 1) * 2, h)

                    # Calculate max for potentially smaller last block
                    mipmap[i, j] = np.max(z[i * 2:i_end, j * 2:j_end])

            result[0:w_, y0:y0 + h_] = mipmap
            z = mipmap
            y0 += h_
            level += 1

        # Final level: single maximum value over entire remaining z
        max_value = np.max(z)
        result[0, y0] = max_value

        return result, level

    def get_value(self, i, j, level):
        # Check if the level is valid
        if level < 0:
            raise ValueError("Level must be non-negative")
        elif level == 0:
            return self.z
        elif level > self.levels - 1:
            raise ValueError(f"Level exceeded maximum ({level} > {self.levels - 1})")
        else:
            level_ = level - 1
            # Get the shape of maxmipmap to determine how many levels are stored
            w, h = self.maxmipmap.shape

            # Determine the width and height of the requested level
            current_width = w
            current_height = h
            y_offset = 0

            for i in range(level_):
                if current_width == 1 or current_height == 1:
                    break  # We've reached the smallest possible dimension for this direction
                current_width = current_width // 2 + (1 if current_width % 2 != 0 else 0)
                current_height = current_height // 2 + (1 if current_height % 2 != 0 else 0)
                y_offset += current_height

            # Extract the level
            if current_width > 0 and current_height > 0:
                return self.maxmipmap[:current_width, y_offset:y_offset + current_height // 2]
            else:
                # If we've gone beyond the data, return the last level (which should be 1x1 or similar)
                return self.maxmipmap[:1, y_offset:y_offset + 1]
