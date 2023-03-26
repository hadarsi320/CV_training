import numpy as np


def rotate_around(points, angle, rotate_center):
    normalized_points = points - rotate_center
    rotated_points = rotate_points(normalized_points, angle) + rotate_center
    return rotated_points


def rotate_points(points, angle):
    radian_angle = angle * np.pi / 180
    rotate_matrix = np.array([
        [np.cos(radian_angle), np.sin(radian_angle)],
        [-np.sin(radian_angle), np.cos(radian_angle)]
    ])
    rotated_points = points @ rotate_matrix
    return rotated_points


def scale_around(points, scale, scale_center):
    normalized_points = points - scale_center
    scaled_points = normalized_points * scale + scale_center
    return scaled_points
