import cv2 as cv

from basic_shape import BasicShape

__all__ = ['Circle']


class Circle(BasicShape):
    def __init__(self, p, radius, **kwargs):
        super(Circle, self).__init__(**kwargs)
        self.points = [p]
        self.radius = radius

    def draw(self, canvas):
        if self.line_color:
            cv.circle(canvas, self.points[0], self.radius, self.fill_color, 1)
        if self.fill_color:
            cv.circle(canvas, self.points[0], self.radius, self.fill_color, -1)

    def scale(self, scale):
        self.radius *= scale
