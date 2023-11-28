import numpy as np
import cv2

patch_size = 5

def get_patch(im, x, y, patch_size):
    x, y = int(np.floor(x)), int(np.floor(y))
    y_max, x_max = im.shape[:2]
    xd, xt = max(0, x-patch_size), min(x+patch_size, x_max)
    yd, yt = max(0, y-patch_size), min(y+patch_size, y_max)
    return im[yd:yt, xd:xt]


def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    xl, yl = pts1[0, 0], pts1[0, 1]
    l = np.dot(np.array([xl, yl, 1]), F.T)

    yr_max, xr_max = im2.shape[:2]
    if l[0] < l[1]:
        yr = np.array(list(range(yr_max)))
        xr = -(l[1] * yr + l[2]) / l[0]
    else:
        xr = np.array(list(range(xr_max)))
        yr = -(l[0] * xr + l[2]) / l[1]
    xr, yr = xr[xr>=patch_size], yr[xr>=patch_size]
    xr, yr = xr[yr>=patch_size], yr[yr>=patch_size]
    xr, yr = xr[xr<xr_max-patch_size], yr[xr<xr_max-patch_size]
    xr, yr = xr[yr<yr_max-patch_size], yr[yr<yr_max-patch_size]
    patch_l = get_patch(im1, xl, yl, patch_size)
    min_diff, x_best, y_best = np.sum(np.abs(patch_l-get_patch(im2, xr[0], yr[0], patch_size))), xr[0], yr[0]
    for x, y in zip(xr[1:], yr[1:]):
        diff = np.sum(np.abs(patch_l-get_patch(im2, x, y, patch_size)))
        if diff < min_diff:
            min_diff = diff
            x_best = x
            y_best = y
    return np.array([[x_best, y_best]])
