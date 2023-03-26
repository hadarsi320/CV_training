import cv2 as cv

from basic_shape import BasicShape

__all__ = ['Line']


class Line(BasicShape):
    def __init__(self, p1, p2, **kwargs):
        super(Line, self).__init__(**kwargs)
        if self.line_color is None:
            raise ValueError('Line color must be passed to line shape')
        self.p1 = p1
        self.p2 = p2

    def draw(self, canvas):
        cv.line(canvas, self.p1, self.p2, self.line_color, 1)
