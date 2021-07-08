import math

import numba
import numpy as np
from numba import cuda, prange
from PIL import Image as im


def generate_julia(width=1200, height=800, max_iterations=50, cx=-0.7, cy=0.27015, color_hue=204, color_saturation=0.64,
                   color_intensity=1.0, use_gpu=True):
    pixels = np.zeros([width, height, 3], dtype=np.uint8)

    if use_gpu:
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(pixels.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(pixels.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        __julia_cuda[blockspergrid, threadsperblock](pixels, width, height, max_iterations, cx, cy, color_hue,
                                                     color_saturation, color_intensity)
    else:
        __julia_cpu(pixels, width, height, max_iterations, cx, cy, color_hue, color_saturation, color_intensity)

    return pixels, im.fromarray(pixels.transpose((1, 0, 2)), 'HSV').convert('RGB')


@numba.jit(nopython=True, parallel=True)
def __julia_cpu(pixels, width, height, max_iterations, cx, cy, color_hue, color_saturation, color_intensity):
    for x in prange(0, width):
        for y in prange(0, height):
            c = complex(cx, cy)
            z = complex(1.5 * (x - width / 2) / (0.5 * width), 1.0 * (y - height / 2) / (0.5 * height))

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
def __julia_cuda(pixels, width, height, max_iterations, cx, cy, color_hue, color_saturation, color_intensity):
    x, y = cuda.grid(2)

    if x < pixels.shape[0] and y < pixels.shape[1]:
        c = complex(cx, cy)
        z = complex(1.5 * (x - width / 2) / (0.5 * width), 1.0 * (y - height / 2) / (0.5 * height))

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
