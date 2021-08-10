from dataclasses import dataclass

import numpy as np
from PIL import Image as im

@dataclass
class Plane2d:
    width: int = 1920
    height: int = 1080


@dataclass
class ComplexPlane:
    real_begin: float = -2.0
    real_end: float = 1.0
    imag_begin: float = -2.0
    imag_end: float = 1.0


@dataclass
class HsvColor:
    hue: int = 204
    saturation: float = 0.64
    intensity: float = 2.0


def image_rgb_from_hsv(pixels: np.array) -> im:
    """
    Create RGB Pillow image from numpy array with HSV pixels

    Args:
        pixels: 2D array of HSV pixels

    Returns:
        Pillow RGB image
    """

    return im.fromarray(pixels.transpose((1, 0, 2)), 'HSV').convert('RGB')