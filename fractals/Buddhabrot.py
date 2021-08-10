import numpy as np
from numba.cuda.random import create_xoroshiro128p_states

from fractals.MandelbrotBase import MandelbrotBase
from fractals.kernels.buddhabrot import buddhabrot, buddhabrot_cuda, draw_buddhabrot


class Buddhabrot(MandelbrotBase):
    def __init__(self, plane, complex_plane, max_iterations, hsv_color):
        super().__init__(plane, complex_plane, max_iterations, hsv_color)

    def compute(self, total_samples=10000000):
        counters = np.zeros([self._plane.width, self._plane.height], dtype=np.uint16)

        print("Computing buddhabrot...")
        buddhabrot(counters, self._plane.width, self._plane.height, self._max_iterations, total_samples,
                   self._complex_plane.real_begin, self._complex_plane.real_end, self._complex_plane.imag_begin,
                   self._complex_plane.imag_end)

        pixels = np.zeros([self._plane.width, self._plane.height, 3], dtype=np.uint8)
        print("Drawing buddhabrot...")
        draw_buddhabrot(pixels, counters, self._plane.width, self._plane.height, self._hsv_color.hue,
                        self._hsv_color.saturation, self._hsv_color.intensity)

        return pixels

    def compute_gpu(self, samples_per_thread=128):
        counters = np.zeros([self._plane.width, self._plane.height], dtype=np.uint16)

        threads_per_block = 256
        total_blocks = 2048
        rng_states = create_xoroshiro128p_states(threads_per_block * total_blocks, seed=3123)

        print("Computing buddhabrot...")
        buddhabrot_cuda[total_blocks, threads_per_block](counters, rng_states, self._plane.width,
                                                         self._plane.height, self._max_iterations,
                                                         samples_per_thread, self._complex_plane.real_begin,
                                                         self._complex_plane.real_end,
                                                         self._complex_plane.imag_begin,
                                                         self._complex_plane.imag_end)

        pixels = np.zeros([self._plane.width, self._plane.height, 3], dtype=np.uint8)

        print("Drawing buddhabrot...")
        draw_buddhabrot(pixels, counters, self._plane.width, self._plane.height, self._hsv_color.hue,
                        self._hsv_color.saturation, self._hsv_color.intensity)

        return pixels
