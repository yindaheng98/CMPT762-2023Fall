import numpy as np
from itertools import product
from epipolarCorrespondence import get_patch, diff_patches


def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and
    im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.
    """
    dispM = np.zeros_like(im1, dtype=float)
    w = (windowSize - 1) // 2
    y1_max, x1_max = im1.shape[:2]
    y2_max, x2_max = im2.shape[:2]
    for (x1, y) in product(range(x1_max), range(min(y1_max, y2_max))):
        patch_l = get_patch(im1, x1, y, w)
        patch_r = np.array([get_patch(im1, x2, y, w) for x2 in range(max(0, x1 - maxDisp), min(x1 + maxDisp, x2_max))])
        diff = diff_patches(patch_l, patch_r)
        dispM[y, x1] = x1 - np.argmin(diff)
    return dispM
