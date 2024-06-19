import cv2
import numpy as np
from tqdm import trange


def disparity_ssd(L: np.ndarray, R: np.ndarray, w=31, d_max=100):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    assert w % 2 == 1
    # inefficient version - sliding window
    """
    n_rows, n_cols = L.shape
    D = np.empty_like(L)
    pad_L = np.pad(L, w // 2)
    pad_R = np.pad(R, w // 2)

    L_windows = np.lib.stride_tricks.sliding_window_view(pad_L, (w, w))
    R_windows = np.lib.stride_tricks.sliding_window_view(pad_R, (w, w))

    L_windows = L_windows[:, :, np.newaxis]
    R_windows = R_windows[:, np.newaxis, :]
    for i in trange(n_rows):
        ssd_mat = np.square(L_windows[i] - R_windows[i]).sum((-2, -1))
        matches = np.argmin(ssd_mat, -1)
        disparities = matches - np.arange(n_cols)
        D[i] = disparities
    """

    # efficient version - sharing computations
    # pad_L = np.pad(L, w // 2)
    # pad_R = np.pad(R, w // 2)

    ssd_matrices = np.empty((2 * d_max + 1, *L.shape))
    n_cols = L.shape[1]

    for i in trange(-d_max, d_max + 1):
        shifted_R = np.roll(R, i, axis=1)
        if i > 0:
            shifted_R[:, :i] = 0
        else:
            shifted_R[:, i:] = 0
        ssd = cv2.blur(np.square(L - shifted_R), (w, w))
        ssd_matrices[i + d_max] = ssd

    D = np.empty_like(L)
    for j in trange(n_cols):
        start = max(d_max - j, 0)
        end = j - n_cols - 1 if n_cols - j < d_max else -1
        rel_ssd_matrices = ssd_matrices[start:end, :, j]
        D[:, j] = np.argmin(rel_ssd_matrices, axis=0)
    return D
