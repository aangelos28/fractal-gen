"""
Methods for generating the mandelbrot fractal.
"""

import math

import numba
import numpy as np
from numba import cuda, prange
from PIL import Image as im


def generate_mandelbrot(width=1200, height=800, max_iterations=50, re_start=-2.0, re_end=1.0, im_start=-1.0, im_end=1.0,
                        color_hue=204, color_saturation=0.64, color_intensity=1.0, use_gpu=True):
    """
    Generate a mandelbrot visualization.

    Args:
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
        use_gpu: Whether to use CUDA to compute the fractal

    Returns:
        Pixels of the visualization and a pillow image
    """

    pixels = np.zeros([width, height, 3], dtype=np.uint8)

    if use_gpu:
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(pixels.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(pixels.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        __mandelbrot_cuda[blockspergrid, threadsperblock](pixels, width, height, max_iterations, re_start, re_end,
                                                          im_start, im_end, color_hue, color_saturation,
                                                          color_intensity)
    else:
        __mandelbrot_cpu(pixels, width, height, max_iterations, re_start, re_end, im_start, im_end, color_hue,
                         color_saturation, color_intensity)

    return pixels, im.fromarray(pixels.transpose((1, 0, 2)), 'HSV').convert('RGB')


@numba.jit(nopython=True, parallel=True)
def __mandelbrot_cpu(pixels, width, height, max_iterations, re_start, re_end, im_start, im_end, color_hue,
                     color_saturation, color_intensity):
    """
    Generate a mandelbrot visualization using multi-threading.

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
        use_gpu: Whether to use CUDA to compute the mandelbrot
    """

    for x in prange(0, width):
        for y in prange(0, height):
            c = complex((re_start + (x / width) * (re_end - re_start)), (im_start + (y / height) * (im_end - im_start)))
            z = 0.0j

            iterations = 0
            while (z.real * z.real + z.imag * z.imag < 4.0) and iterations < max_iterations:
                z = z * z + c
                iterations += 1

            # Color smoothing
            smooth_iterations = iterations - math.log(math.log(z.real * z.real + z.imag * z.imag)) + 4.0

            if iterations >= max_iterations:
                pixels[x, y, 0] = 0
                pixels[x, y, 1] = 0
                pixels[x, y, 2] = 0
            else:
                pixels[x, y, 0] = 255 * (color_hue / 360)
                pixels[x, y, 1] = 255 * color_saturation
                pixels[x, y, 2] = 255 * (color_intensity * smooth_iterations / max_iterations)


@cuda.jit
def __mandelbrot_cuda(pixels, width, height, max_iterations, re_start, re_end, im_start, im_end, color_hue,
                      color_saturation, color_intensity):
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
        use_gpu: Whether to use CUDA to compute the mandelbrot
    """

    x, y = cuda.grid(2)

    if x < pixels.shape[0] and y < pixels.shape[1]:
        c = complex((re_start + (x / width) * (re_end - re_start)), (im_start + (y / height) * (im_end - im_start)))
        z = 0.0j

        iterations = 0
        while (z.real * z.real + z.imag * z.imag < 4.0) and iterations < max_iterations:
            z = z * z + c
            iterations += 1

        # Color smoothing
        smooth_iterations = iterations - math.log2(math.log2(z.real * z.real + z.imag * z.imag)) + 4.0

        if iterations >= max_iterations:
            pixels[x, y, 0] = 0
            pixels[x, y, 1] = 0
            pixels[x, y, 2] = 0
        else:
            pixels[x, y, 0] = 255 * (color_hue / 360)
            pixels[x, y, 1] = 255 * color_saturation
            pixels[x, y, 2] = 255 * (color_intensity * smooth_iterations / max_iterations)
