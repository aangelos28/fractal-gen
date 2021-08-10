from fractals.common import Plane2d, ComplexPlane, HsvColor


class MandelbrotBase:
    @property
    def plane(self) -> Plane2d:
        return self._plane

    @plane.setter
    def plane(self, value):
        self._plane = value

    @property
    def complex_plane(self) -> ComplexPlane:
        return self._complex_plane

    @complex_plane.setter
    def complex_plane(self, value):
        self._complex_plane = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        self._max_iterations = value

    @property
    def hsv_color(self) -> HsvColor:
        return self._hsv_color

    @hsv_color.setter
    def hsv_color(self, value):
        self._hsv_color = value

    def __init__(self, plane, complex_plane, max_iterations, hsv_color):
        self._plane = plane
        self._complex_plane = complex_plane
        self._max_iterations = max_iterations
        self._hsv_color = hsv_color
