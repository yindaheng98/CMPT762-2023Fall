import numpy as np
from epipolarCorrespondence import get_patch


def diff_patches(patch, patches, maxDisp):
    patch_ = patch.reshape((patch.shape[0], -1))
    patch_mean = patch_.mean(axis=1)
    patch_std = patch_.std(axis=1)

    patches_ = patches.reshape((patches.shape[0], -1))
    patches_mean = patches_.mean(axis=1)
    patches_std = patches_.std(axis=1)

    mean_ = (patch_.T - patch_mean).T
    means_ = (patches_.T - patches_mean).T

    w = maxDisp//2
    x_max = min(mean_.shape[0], means_.shape[0])-w
    patches_split = np.array([patches_[x-w:x+w, :] for x in range(w, x_max)])
    means_split = np.array([means_[x-w:x+w, :] for x in range(w, x_max)])
    stds_split = np.array([patches_std[x-w:x+w] for x in range(w, x_max)])

    ediff = patch_[w:x_max] - patches_split.transpose((1, 0, 2))
    edist = np.sum(ediff * ediff, axis=2).T

    mean = np.mean(mean_[w:x_max, :] * means_split.transpose((1, 0, 2)), axis=2)
    std = (patch_std[w:x_max] * stds_split.T)
    std[std < 1e-6] = 1e-6
    corr = (mean / std).T

    return corr


def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and
    im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.
    """
    dispM = np.zeros_like(im1, dtype=float)
    w = (windowSize - 1) // 2
    y1_max, x1_max = im1.shape[:2]
    y2_max, x2_max = im2.shape[:2]
    for y in range(min(y1_max, y2_max)):
        patch_l = np.array([get_patch(im1, x1, y, w) for x1 in range(x1_max)])
        patch_r = np.array([get_patch(im2, x2, y, w) for x2 in range(x2_max)])
        diff = diff_patches(patch_l, patch_r, maxDisp)
        x_min, x_max = maxDisp//2, min(x1_max, x2_max)-maxDisp//2
        dispMline = maxDisp//2 - np.argmax(diff, axis=1)
        dispMline[np.sum(diff, axis=1) < 1e-6] = 0
        dispM[y, x_min:x_max] = dispMline
    return dispM
