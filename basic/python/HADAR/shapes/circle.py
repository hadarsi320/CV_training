import cv2 as cv

from .basic_shape import BasicShape

__all__ = ['Circle']


class Circle(BasicShape):
    def __init__(self, p, radius, **kwargs):
        self.radius = radius
        super(Circle, self).__init__([p], **kwargs)

    def draw(self, canvas):
        if self.fill_color is not None:
            cv.circle(canvas, self.points[0], self.radius, self.fill_color, -1)
        if self.line_color is not None:
            cv.circle(canvas, self.points[0], self.radius, self.line_color, 1)

    def scale(self, scale, center):
        super(Circle, self).scale(scale, center)
        self.radius = round(self.radius * scale)

    def get_bounding_box(self):
        p = self.points[0]
        return {
            'left': p[0] - self.radius,
            'right': p[0] + self.radius,
            'top': p[1] - self.radius,
            'bottom': p[1] + self.radius
        }
