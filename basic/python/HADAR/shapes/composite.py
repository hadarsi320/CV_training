from typing import List

import numpy as np

from .basic_shape import BasicShape
from .shape import Shape

__all__ = ['Composite']


class Composite(Shape):
    def __init__(self, shapes: List[BasicShape], **kwargs):
        self.shapes = shapes
        center = self.get_composite_center()
        super(Composite, self).__init__(center, **kwargs)

    def draw(self, canvas):
        for shape in self.shapes:
            shape.draw(canvas)

    def rotate(self, angle, center):
        for shape in self.shapes:
            shape.rotate(angle, center)

    def translate(self, translation):
        for shape in self.shapes:
            shape.translate(translation)

    def scale(self, scale, center):
        for shape in self.shapes:
            shape.scale(scale, center)

    def get_bounding_box(self):
        bounding_boxes = [shape.get_bounding_box() for shape in self.shapes]
        comp_bb = {
            'left': min(bb['left'] for bb in bounding_boxes),
            'right': max(bb['right'] for bb in bounding_boxes),
            'top': min(bb['top'] for bb in bounding_boxes),
            'bottom': max(bb['bottom'] for bb in bounding_boxes)
        }

        return comp_bb

    def get_composite_center(self):
        bb = self.get_bounding_box()
        center = np.array([
            (bb['left'] + bb['right']) / 2,
            (bb['top'] + bb['bottom']) / 2
        ])
        return center

    def get_point_mean_center(self):
        points = []
        for shape in self.shapes:
            points.append(shape.points)
        center = np.concatenate(points).mean(axis=0)
        return center
