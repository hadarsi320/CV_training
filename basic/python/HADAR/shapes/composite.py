import numpy as np
from typing import List

from .shape import Shape
from .basic_shape import BasicShape
from .shape_utils import rotate_points

__all__ = ['Composite']


def get_center(shapes):
    points = []
    for shape in shapes:
        points.append(shape.points)
    center = np.concatenate(points).mean(axis=0)
    return center


class Composite(Shape):
    def __init__(self, shapes: List[BasicShape], center: List = None, **kwargs):
        self.shapes = shapes
        if center:
            center = np.array(center)
        else:
            center = get_center(shapes)
        super(Composite, self).__init__(center, **kwargs)

    def draw(self, canvas):
        for shape in self.shapes:
            shape.draw(canvas)

    def rotate(self, angle, center):
        for shape in self.shapes:
            shape.rotate(angle, center)

    def translate(self, translation):
        for shape in self.shapes:
            shape.translate(translation)

    def scale(self, scale, center):
        for shape in self.shapes:
            shape.scale(scale, center)
