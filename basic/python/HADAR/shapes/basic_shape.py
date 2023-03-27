from abc import ABC

import cv2 as cv

from .shape import Shape
from .shape_utils import *


class BasicShape(Shape, ABC):
    def __init__(self, points, line_color=None, fill_color=None, **kwargs):
        assert line_color or fill_color, 'Either fill color or line color must be passed'
        self.points = np.array([np.array(p) for p in points], dtype=float)
        self.line_color = preprocess_color(line_color)
        self.fill_color = preprocess_color(fill_color)

        center = self.points.mean(axis=0)
        super(BasicShape, self).__init__(center, **kwargs)

    def draw(self, canvas):
        assert len(self.points) > 2
        points = [self.points.reshape((-1, 1, 2)).astype(np.int32)]
        if self.fill_color is not None:
            cv.fillPoly(canvas, points, self.fill_color)
        if self.line_color is not None:
            cv.polylines(canvas, points, True, self.line_color)

    def translate(self, translation):
        translation = np.array(translation)
        for point in self.points:
            point += translation
        self.points = self.points

    def rotate(self, angle, center):
        rotated_points = rotate_around(self.points, angle, center)
        self.points = rotated_points

    def scale(self, scale, center):
        scaled_points = scale_around(self.points, scale, center)
        self.points = scaled_points

    def get_bounding_box(self):
        return {
            'left': self.points.min(axis=0)[0],
            'right': self.points.max(axis=0)[0],
            'top': self.points.min(axis=0)[1],
            'bottom': self.points.max(axis=0)[1]
        }


def preprocess_color(color):
    if color is None:
        return None
    color = np.array(color)[::-1]  # because color in opencv is in GBR format
    color = color / 255  # because opencv expects color in the range of [0,1]
    return color
