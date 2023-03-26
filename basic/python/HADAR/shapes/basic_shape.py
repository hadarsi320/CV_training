import warnings
from abc import ABC

import numpy as np
import cv2 as cv

from .shape import Shape
from .shape_utils import rotate_points


class BasicShape(Shape, ABC):
    def __init__(self, points, line_color=None, fill_color=None):
        assert line_color or fill_color, 'Either fill color or line color must be passed'
        self.points = np.array([np.array(p) for p in points], dtype=np.int32)
        self.line_color = line_color
        self.fill_color = fill_color

    def draw(self, canvas):
        assert len(self.points) > 2
        points = [self.points.reshape((-1, 1, 2))]
        if self.fill_color:
            cv.fillPoly(canvas, points, self.fill_color)
        if self.line_color:
            cv.polylines(canvas, points, True, self.line_color)

    def translate(self, translation):
        translation = np.array(translation)
        for point in self.points:
            point += translation
        self.points = np.round(self.points)

    def rotate(self, angle):
        if len(self.points) > 1:
            mean = self.points.mean(axis=0)
            normalized_points = self.points - mean
            rotated_points = rotate_points(normalized_points, angle)
            self.points = np.round(rotated_points + mean).astype(np.int32)
        else:
            warnings.warn('Object only has a single point and so rotation does nothing')

    def scale(self, scale):
        mean = self.points.mean(axis=0)
        normalized_points = self.points - mean
        scaled_points = normalized_points * scale
        self.points = np.round(scaled_points + mean).astype(np.int32)
