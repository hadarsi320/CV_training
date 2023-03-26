import cv2 as cv
import numpy as np

from basic_shape import BasicShape

__all__ = ['Rectangle']


class Rectangle(BasicShape):
    def __init__(self, top_left, bottom_right, **kwargs):
        super(Rectangle, self).__init__(**kwargs)
        assert all(top_left < bottom_right), 'Points given must be top left and bottom right'
        bottom_left = np.array([top_left[0], bottom_right[1]])
        top_right = np.array([bottom_right[0], top_left[1]])
        self.points = [top_left, top_right, bottom_right, bottom_left]

    def draw(self, canvas):
        if self.line_color:
            cv.polylines(canvas, self.points, False, self.line_color)
        if self.fill_color:
            cv.polylines(canvas, self.points, True, self.fill_color)
