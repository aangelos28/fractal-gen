from dataclasses import dataclass


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
