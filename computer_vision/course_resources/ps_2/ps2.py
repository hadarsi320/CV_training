# ps2
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr

plt.set_cmap('gray')


def imsave(image: np.ndarray, path):
    norm_image = ((image + image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    cv2.imwrite(path, norm_image)

    # plt.imshow(image)
    # plt.colorbar()
    # plt.savefig('output/ps2-1-a-1.png')
    # plt.close()


def part1():
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

    D_L = disparity_ssd(L, R)
    D_R = disparity_ssd(R, L)

    imsave(D_L, 'output/ps2-1-a-1.png')
    imsave(D_R, 'output/ps2-1-a-2.png')


def part2():
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)

    D_L = disparity_ncorr(L, R)
    D_R = disparity_ncorr(R, L)

    imsave(D_L, 'output/ps2-2-a-1.png')
    imsave(D_R, 'output/ps2-2-a-2.png')


if __name__ == '__main__':
    # part1()
    # part2()

    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)

    D_L = disparity_ssd(L, R)
    D_R = disparity_ssd(R, L)

    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.imshow(D_L)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(D_R)
    plt.colorbar()
    plt.show()

