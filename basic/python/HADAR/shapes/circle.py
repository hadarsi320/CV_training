import cv2 as cv

from .basic_shape import BasicShape

__all__ = ['Circle']


class Circle(BasicShape):
    def __init__(self, p, radius, **kwargs):
        self.radius = radius
        super(Circle, self).__init__([p], **kwargs)

    def draw(self, canvas):
        if self.fill_color:
            cv.circle(canvas, self.points[0], self.radius, self.fill_color, -1)
        if self.line_color:
            cv.circle(canvas, self.points[0], self.radius, self.line_color, 1)

    def scale(self, scale, scale_center):
        super(Circle, self).scale(scale, scale_center)
        self.radius = round(self.radius * scale)
