import numpy as np
import cv2 as cv

"""
This would be a singleton if it wasn't such a pain in the ass 
"""


class Drawer:
    def __init__(self):
        self.canvas = None

    def draw(self, shapes, canvas_size, canvas_color=(0, 0, 0)):
        self.canvas = np.ones((*canvas_size, 3)) * canvas_color
        for shape in shapes:
            shape.draw(self.canvas)

    def show(self):
        if self.canvas is None:
            raise ValueError('Show called before canvas drawn')
        cv.imshow("Floopy", self.canvas)
        cv.waitKey(0)

    def save(self, save_file):
        if self.canvas is None:
            raise ValueError('Save called before canvas drawn')
        cv.imwrite(save_file, (self.canvas * 255).astype(int))
