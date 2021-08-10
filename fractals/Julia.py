import math

import numpy as np

from fractals.MandelbrotBase import MandelbrotBase
from fractals.kernels.julia import julia_cuda, julia


class Julia(MandelbrotBase):
    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, value):
        self._cx = value

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, value):
        self._cy = value

    def __init__(self, plane, complex_plane, max_iterations, hsv_color, cx, cy):
        super().__init__(plane, complex_plane, max_iterations, hsv_color)
        self._cx = cx
        self._cy = cy

    def compute(self, use_gpu=True):
        pixels = np.zeros([self._plane.width, self._plane.height, 3], dtype=np.uint8)

        if use_gpu:
            threads_per_block = (16, 16)
            blocks_x = math.ceil(pixels.shape[0] / threads_per_block[0])
            blocks_y = math.ceil(pixels.shape[1] / threads_per_block[1])
            blocks_in_grid = (blocks_x, blocks_y)

            julia_cuda[blocks_in_grid, threads_per_block](pixels, self._plane.width, self._plane.height,
                                                          self._max_iterations,
                                                          self._complex_plane.real_begin,
                                                          self._complex_plane.real_end,
                                                          self._complex_plane.imag_begin,
                                                          self._complex_plane.imag_end,
                                                          self._cx, self._cy,
                                                          self._hsv_color.hue,
                                                          self._hsv_color.saturation,
                                                          self._hsv_color.intensity)
        else:
            julia(pixels, self.plane.width, self.plane.height, self.max_iterations,
                  self._complex_plane.real_begin,
                  self._complex_plane.real_end,
                  self._complex_plane.imag_begin,
                  self._complex_plane.imag_end,
                  self.cx, self.cy,
                  self.hsv_color.hue, self.hsv_color.saturation, self.hsv_color.intensity)

        return pixels
