from .basic_shape import BasicShape

__all__ = ['Triangle']


class Triangle(BasicShape):
    def __init__(self, p1, p2, p3, **kwargs):
        super(Triangle, self).__init__([p1, p2, p3], **kwargs)
