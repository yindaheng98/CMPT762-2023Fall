import numpy as np
import cv2

patch_size = 3


def get_patch(im, x, y, patch_size):
    x, y = int(np.floor(x)), int(np.floor(y))
    y_max, x_max = im.shape[:2]
    xd, xt = max(0, x-patch_size), min(x+patch_size, x_max)
    yd, yt = max(0, y-patch_size), min(y+patch_size, y_max)
    patch = np.zeros([patch_size * 2 + 1, patch_size * 2 + 1, *im.shape[2:]])
    xd_, xt_ = max(patch_size-x, 0), patch_size + min(x_max-x, patch_size)
    yd_, yt_ = max(patch_size-y, 0), patch_size + min(y_max-y, patch_size)
    patch[yd_:yt_, xd_:xt_, ...] = im[yd:yt, xd:xt, ...]
    return patch


def diff_patches(patch, patches):
    patch_ = patch.reshape(-1)
    patch_mean = patch_.mean()
    patch_std = patch_.std()

    patches_ = patches.reshape((patches.shape[0], -1))
    patches_mean = patches_.mean(axis=1)
    patches_std = patches_.std(axis=1)
    patches_std[patches_std < 1e-6] = 1e-6

    mean = np.mean((patch_ - patch_mean) * (patches_.T - patches_mean).T, axis=1)
    std = patch_std * patches_std
    std[std < 1e-6] = 1e-6

    return mean / std


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
    if l[0] > l[1]:
        yr = np.array(list(range(yr_max)))
        xr = -(l[1] * yr + l[2]) / l[0]
    else:
        xr = np.array(list(range(xr_max)))
        yr = -(l[0] * xr + l[2]) / l[1]
    xr, yr = xr[xr >= patch_size], yr[xr >= patch_size]
    xr, yr = xr[yr >= patch_size], yr[yr >= patch_size]
    xr, yr = xr[xr < xr_max-patch_size], yr[xr < xr_max-patch_size]
    xr, yr = xr[yr < yr_max-patch_size], yr[yr < yr_max-patch_size]

    im1_YCrCb, im2_YCrCb = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb), cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    patch_l = get_patch(im1_YCrCb, xl, yl, patch_size)
    patch_r = np.array([get_patch(im2_YCrCb, x, y, patch_size) for x, y in zip(xr, yr)])
    diff = diff_patches(patch_l, patch_r)
    return np.array([[xr[np.argmax(diff)], yr[np.argmax(diff)]]])
