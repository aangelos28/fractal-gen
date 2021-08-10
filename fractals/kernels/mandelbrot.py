import math

import numba
from numba import cuda, prange


@numba.jit(nopython=True, parallel=True)
def mandelbrot(pixels, width, height, max_iterations, re_start, re_end, im_start, im_end, color_hue,
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
    """

    for x in prange(0, width):
        for y in prange(0, height):
            c = complex((re_start + (x / width) * (re_end - re_start)),
                        (im_start + (y / height) * (im_end - im_start)))
            z = 0.0j

            iterations = 0
            while (abs(z) < 4.0) and iterations < max_iterations:
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
                pixels[x, y, 2] = 255 * min(color_intensity * smooth_iterations / max_iterations, 1)


@cuda.jit
def mandelbrot_cuda(pixels, width, height, max_iterations, re_start, re_end, im_start, im_end, color_hue,
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
    """

    x, y = cuda.grid(2)

    if x < pixels.shape[0] and y < pixels.shape[1]:
        c = complex((re_start + (x / width) * (re_end - re_start)), (im_start + (y / height) * (im_end - im_start)))
        z = 0.0j

        iterations = 0
        while (abs(z) < 4.0) and iterations < max_iterations:
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
            pixels[x, y, 2] = 255 * min(color_intensity * smooth_iterations / max_iterations, 1)
