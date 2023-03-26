import warnings

import cv2 as cv

from basic_shape import BasicShape

__all__ = ['Point']


class Point(BasicShape):
    def __init__(self, p, **kwargs):
        super(Point, self).__init__(**kwargs)
        self.points = [p]

    def draw(self, canvas):
        cv.circle(canvas, self.points[0], 1, self.fill_color)

    def scale(self, scale):
        warnings.warn('Scaling does nothing on a point')
