from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, center, translation=None, rotate=None, scale=None):
        if translation:
            self.translate(translation)
            center += translation
        if rotate:
            self.rotate(rotate, center)
        if scale:
            self.scale(scale, center)

    @abstractmethod
    def draw(self, canvas):
        pass

    @abstractmethod
    def rotate(self, angle, rotate_center):
        pass

    @abstractmethod
    def translate(self, translation):
        pass

    @abstractmethod
    def scale(self, scale, scale_center):
        pass
