import warnings

import cv2 as cv
import numpy as np

from .basic_shape import BasicShape

__all__ = ['Line']


class Line(BasicShape):
    def __init__(self, p1, p2, **kwargs):
        super(Line, self).__init__([p1, p2], **kwargs)
        if self.line_color is None:
            raise ValueError('Line color must be passed to lines')
        if self.fill_color is not None:
            warnings.warn('Fill color is meaningless for lines')

    def draw(self, canvas):
        cv.line(canvas, *self.points, self.line_color, 1)
