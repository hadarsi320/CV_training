from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, center, translation=None, rotate=None, scale=None):
        if translation:
            self.translate(translation)
            center += translation
        if scale:
            self.scale(scale, center)
        if rotate:
            self.rotate(rotate, center)

    @abstractmethod
    def draw(self, canvas):
        pass

    @abstractmethod
    def rotate(self, angle, center):
        pass

    @abstractmethod
    def translate(self, translation):
        pass

    @abstractmethod
    def scale(self, scale, center):
        pass

    @abstractmethod
    def get_bounding_box(self):
        pass
