import cv2 as cv
import numpy as np

from .basic_shape import BasicShape

__all__ = ['Circle']


class Circle(BasicShape):
    def __init__(self, p, radius, **kwargs):
        self.radius = radius
        super(Circle, self).__init__([p], **kwargs)

    def draw(self, canvas):
        p = self.points[0].astype(np.int32)
        r = round(self.radius)
        if self.fill_color is not None:
            cv.circle(canvas, p, r, self.fill_color, -1)
        if self.line_color is not None:
            cv.circle(canvas, p, r, self.line_color, 1)

    def scale(self, scale, center):
        super(Circle, self).scale(scale, center)
        self.radius = self.radius * scale

    def get_bounding_box(self):
        p = self.points[0]
        return {
            'left': p[0] - self.radius,
            'right': p[0] + self.radius,
            'top': p[1] - self.radius,
            'bottom': p[1] + self.radius
        }
