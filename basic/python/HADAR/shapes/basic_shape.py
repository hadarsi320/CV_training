import warnings
from abc import ABC

from shape import Shape


class BasicShape(Shape, ABC):
    def __init__(self, points, line_color=None, fill_color=None):
        assert line_color or fill_color, 'Either fill color or line color must be passed'
        self.points = points
        self.line_color = line_color
        self.fill_color = fill_color

    def translate(self, translation):
        for point in self.points:
            point += translation

    def rotate(self, angle):
        if len(self.points) > 1:
            # todo implement
            pass
        else:
            warnings.warn('Object only has a single point and so rotation does nothing')

    def scale(self, scale):
        # todo implement
        pass
