from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def draw(self, canvas):
        pass

    @abstractmethod
    def rotate(self, angle):
        pass

    @abstractmethod
    def translate(self, translation):
        pass

    @abstractmethod
    def scale(self, scale):
        pass
