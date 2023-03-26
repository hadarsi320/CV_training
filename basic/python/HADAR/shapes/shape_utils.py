import numpy as np


def rotate_points(points, angle):
    radian_angle = angle * np.pi / 180
    rotate_matrix = np.array([
        [np.cos(radian_angle), np.sin(radian_angle)],
        [-np.sin(radian_angle), np.cos(radian_angle)]
    ])
    rotated_points = points @ rotate_matrix
    return rotated_points
