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
    def __init__(self, shapes: List[BasicShape], center: List = None):
        self.shapes = shapes
        if center:
            self.center = np.array(center)
        else:
            self.center = get_center(shapes)

    def draw(self, canvas):
        for shape in self.shapes:
            shape.draw(canvas)

    def rotate(self, angle):
        for shape in self.shapes:
            points = shape.points
            normalized_points = points - self.center
            rotated_points = rotate_points(normalized_points, angle)
            shape.points = np.round(rotated_points + self.center).astype(np.int32)

    def translate(self, translation):
        for shape in self.shapes:
            shape.translate(translation)
        self.center += translation

    def scale(self, scale):
        for shape in self.shapes:
            shape.scale(scale)
            points = shape.points
            shape_center = points.mean(axis=0)
            shape_bias = shape_center - self.center
            centered_points = points - shape_bias
            scaled_biased_points = centered_points + shape_bias * scale
            shape.points = np.round(scaled_biased_points).astype(np.int32)
