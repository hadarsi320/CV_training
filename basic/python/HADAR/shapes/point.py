import warnings

from .circle import Circle

__all__ = ['Point']


class Point(Circle):
    def __init__(self, p, **kwargs):
        super(Point, self).__init__([p], radius=1, fill_color=None, **kwargs)
        if self.fill_color is not None:
            warnings.warn('Fill color is meaningless for points')

    def scale(self, scale, center):
        warnings.warn('Scaling does nothing on a point')
