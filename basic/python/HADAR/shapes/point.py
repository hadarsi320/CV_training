import warnings

import cv2 as cv

from .basic_shape import BasicShape

__all__ = ['Point']


class Point(BasicShape):
    def __init__(self, p, **kwargs):
        super(Point, self).__init__([p], **kwargs)
        #TODO: why not inherit from circle?
        if self.line_color is None: # TODO: again default?
            raise ValueError('Line color must be passed to points')
        if self.fill_color is not None:
            warnings.warn('Fill color is meaningless for points')

    def draw(self, canvas):
        cv.circle(canvas, self.points[0], 1, self.line_color)

    def scale(self, scale, center):
        warnings.warn('Scaling does nothing on a point')
