import math

import numpy as np
from PIL import Image as im

from fractals.MandelbrotBase import MandelbrotBase
from fractals.kernels.mandelbrot import mandelbrot_cuda, mandelbrot


class Mandelbrot(MandelbrotBase):
    def __init__(self, plane, complex_plane, max_iterations, hsv_color):
        super().__init__(plane, complex_plane, max_iterations, hsv_color)

    def compute(self, use_gpu=True) -> im:
        pixels = np.zeros([self._plane.width, self._plane.height, 3], dtype=np.uint8)

        if use_gpu:
            threads_per_block = (16, 16)
            blocks_x = math.ceil(pixels.shape[0] / threads_per_block[0])
            blocks_y = math.ceil(pixels.shape[1] / threads_per_block[1])
            blocks_in_grid = (blocks_x, blocks_y)

            mandelbrot_cuda[blocks_in_grid, threads_per_block](pixels, self._plane.width, self._plane.height,
                                                               self._max_iterations,
                                                               self._complex_plane.real_begin,
                                                               self._complex_plane.real_end,
                                                               self._complex_plane.imag_begin,
                                                               self._complex_plane.imag_end,
                                                               self._hsv_color.hue,
                                                               self._hsv_color.saturation,
                                                               self._hsv_color.intensity)
        else:
            mandelbrot(pixels, self._plane.width, self._plane.height, self._max_iterations,
                       self._complex_plane.real_begin, self._complex_plane.real_end,
                       self._complex_plane.imag_begin, self._complex_plane.imag_end,
                       self._hsv_color.hue, self._hsv_color.saturation, self._hsv_color.intensity)

        return im.fromarray(pixels.transpose((1, 0, 2)), 'HSV').convert('RGB')
