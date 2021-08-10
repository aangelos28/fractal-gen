from random import uniform as randuniform

import numba
import numpy as np
from numba import cuda, prange
from numba.cuda.random import xoroshiro128p_uniform_float32


@numba.jit(nopython=True)
def __check_sample_trajectory_escapes(sample_real, sample_imag, max_iterations):
    c = complex(sample_real, sample_imag)
    z = 0.0j

    iterations = 0
    while (abs(z) < 10.0) and iterations < max_iterations:
        z = z * z + c
        iterations += 1

    return iterations


@numba.jit(nopython=True)
def __trace_sample_trajectory(counters, sample_real, sample_imag, width, height, re_start, re_end,
                              im_start, im_end):
    c = complex(sample_real, sample_imag)
    z = 0.0j

    while abs(z) < 10.0:
        z = z * z + c

        x = int((z.real - re_start) / ((re_end - re_start) / width))
        y = int((z.imag - im_start) / ((im_end - im_start) / height))

        if (0 < x < counters.shape[0]) and (0 < y < counters.shape[1]):
            counters[x, y] += 1


@numba.jit(nopython=True, parallel=True)
def buddhabrot(counters, width, height, max_iterations, total_samples, re_start, re_end,
               im_start, im_end):
    """
    Generate a mandelbrot visualization using CUDA.

    Args:
        pixels: Reference to the RGB pixel array
        width: Width of the image in pixels
        height: Height of the image in pixels
        max_iterations: Max iterations for orbital escape
        re_start: Minimum value of the real complex plane
        re_end: Maximum value of the real complex plane
        im_start: Minimum value of the imaginary complex plane
        im_end: Maximum value of the imaginary complex plane
        color_hue: Hue of the color used for the visualization
        color_saturation: Saturation of the color used for the visualization
        color_intensity: Intensity of the color used for the visualization
    """

    for _ in prange(0, total_samples):
        # Get random point (sample) in complex plane
        sample_real = randuniform(0, 1) * (re_end - re_start) + re_start
        sample_imag = randuniform(0, 1) * (im_end - im_start) + im_start

        # TODO main cardioid and main bulb optimization

        # Check whether sample escapes, and if so trace its iteration trajectory
        iterations = __check_sample_trajectory_escapes(sample_real, sample_imag, max_iterations)
        if 20 < iterations < max_iterations:
            __trace_sample_trajectory(counters, sample_real, sample_imag, width, height, re_start, re_end, im_start,
                                      im_end)


###################################################################################################################

@cuda.jit(device=True, inline=True)
def __check_sample_trajectory_escapes_cuda(sample_real, sample_imag, max_iterations):
    c = complex(sample_real, sample_imag)
    z = 0.0j

    iterations = 0
    while (abs(z) < 10.0) and iterations < max_iterations:
        z = z * z + c
        iterations += 1

    return iterations


@cuda.jit(device=True, inline=True)
def __trace_sample_trajectory_cuda(counters, sample_real, sample_imag, width, height, re_start, re_end,
                                   im_start, im_end):
    c = complex(sample_real, sample_imag)
    z = 0.0j

    while abs(z) < 10.0:
        z = z * z + c

        x = int((z.real - re_start) / ((re_end - re_start) / width))
        y = int((z.imag - im_start) / ((im_end - im_start) / height))

        if (0 < x < counters.shape[0]) and (0 < y < counters.shape[1]):
            counters[x, y] += 1


@cuda.jit
def buddhabrot_cuda(counters, rng_states, width, height, max_iterations, samples_per_thread, re_start, re_end,
                    im_start, im_end):
    """
    Generate a mandelbrot visualization using CUDA.

    Args:
        pixels: Reference to the RGB pixel array
        width: Width of the image in pixels
        height: Height of the image in pixels
        max_iterations: Max iterations for orbital escape
        re_start: Minimum value of the real complex plane
        re_end: Maximum value of the real complex plane
        im_start: Minimum value of the imaginary complex plane
        im_end: Maximum value of the imaginary complex plane
        color_hue: Hue of the color used for the visualization
        color_saturation: Saturation of the color used for the visualization
        color_intensity: Intensity of the color used for the visualization
    """

    thread_index = cuda.grid(1)

    for i in range(0, samples_per_thread):
        # Get random point (sample) in complex plane
        sample_real = xoroshiro128p_uniform_float32(rng_states, thread_index) * (re_end - re_start) + re_start
        sample_imag = xoroshiro128p_uniform_float32(rng_states, thread_index) * (im_end - im_start) + im_start

        # TODO main cardioid and main bulb optimization

        # Check whether sample escapes, and if so trace its iteration trajectory
        iterations = __check_sample_trajectory_escapes_cuda(sample_real, sample_imag, max_iterations)
        if 20 < iterations < max_iterations:
            __trace_sample_trajectory_cuda(counters, sample_real, sample_imag, width, height, re_start, re_end,
                                           im_start, im_end)


@numba.jit(nopython=True, parallel=True)
def draw_buddhabrot(pixels, counters, width, height, color_hue, color_saturation, color_intensity):
    max_counter = np.amax(counters)

    for x in prange(0, width):
        for y in prange(0, height):
            pixels[x, y, 0] = 255 * (color_hue / 360)
            pixels[x, y, 1] = 255 * color_saturation
            pixels[x, y, 2] = 255 * min(color_intensity * counters[x, y] / max_counter, 1)
