import cv2
import numpy as np
from tqdm import trange


def disparity_ncorr(L, R, w=11):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    assert w % 2 == 1
    n_rows, n_cols = L.shape
    D = np.empty_like(L)
    pad_L = np.pad(L, w // 2).astype(np.float32)
    pad_R = np.pad(R, w // 2).astype(np.float32)

    L_windows = np.lib.stride_tricks.sliding_window_view(pad_L, (w, w))
    R_rows = np.lib.stride_tricks.sliding_window_view(pad_R, w, axis=0).transpose([0, 2, 1])

    for i in trange(n_rows):
        for j in range(n_cols):
            corr_matrix = cv2.matchTemplate(R_rows[i], L_windows[i, j], cv2.TM_CCOEFF_NORMED)[0]
            matches_idx = np.argmax(corr_matrix)
            D[i, j] = matches_idx - j
    return D
