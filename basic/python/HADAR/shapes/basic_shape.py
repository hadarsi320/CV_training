from abc import ABC

import cv2 as cv

from .shape import Shape
from .shape_utils import *


class BasicShape(Shape, ABC):
    def __init__(self, points, line_color=None, fill_color=None, **kwargs):
        assert line_color or fill_color, 'Either fill color or line color must be passed'
        self.points = np.array([np.array(p) for p in points], dtype=np.int32)
        self.line_color = line_color
        self.fill_color = fill_color

        center = self.points.mean(axis=0)
        super(BasicShape, self).__init__(center, **kwargs)

    def draw(self, canvas):
        assert len(self.points) > 2
        points = [self.points.reshape((-1, 1, 2))]
        if self.fill_color:
            cv.fillPoly(canvas, points, self.fill_color)
        if self.line_color:
            cv.polylines(canvas, points, True, self.line_color)

    def translate(self, translation):
        translation = np.array(translation)
        for point in self.points:
            point += translation
        self.points = np.round(self.points)

    def rotate(self, angle, rotate_center):
        rotated_points = rotate_around(self.points, angle, rotate_center).astype(np.int32)
        self.points = rotated_points

    def scale(self, scale, scale_center):
        scaled_points = scale_around(self.points, scale, scale_center).astype(np.int32)
        self.points = scaled_points
