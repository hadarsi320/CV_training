import cv2 as cv

from basic_shape import BasicShape

__all__ = ['Rectangle']


class Rectangle(BasicShape):
    def __init__(self, p1, p2, p3, **kwargs):
        super(Rectangle, self).__init__(**kwargs)
        self.points = [p1, p2, p3]

    def draw(self, canvas):
        if self.line_color:
            cv.polylines(canvas, self.points, False, self.line_color)
        if self.fill_color:
            cv.polylines(canvas, self.points, True, self.fill_color)
