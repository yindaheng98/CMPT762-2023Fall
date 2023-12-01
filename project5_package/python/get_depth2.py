import numpy as np
import cv2
from itertools import product


def Get3dCoord(pts2d, extrinsic, depths):
    n_pts, n_depths = pts2d.shape[0], depths.shape[0]
    K, R, t = extrinsic
    P = np.dot(K, np.concatenate((R.T, [t])).T)
    pts2d_depths = np.stack([pts2d] * depths.shape[0], axis=1) * np.stack([depths] * pts2d.shape[1], axis=1)
    pts2dz = np.expand_dims(np.zeros(pts2d_depths.shape[0:2]) + depths, axis=2)
    pts2dxyz = np.concatenate([pts2d_depths, pts2dz], axis=2)
    pts3d = np.dot(
        pts2dxyz.reshape((n_pts * n_depths, 3)) - P[:, 3],
        np.linalg.inv(P[:, 0:3]).T
    ).reshape((n_pts, n_depths, 3))
    return pts3d


def GetProjCoord(pts3d, extrinsic):
    K, R, t = extrinsic
    n_pts, n_depths = pts3d.shape[0:2]
    pts2d3 = np.dot(np.dot(pts3d.reshape((n_pts * n_depths, 3)), R.T) + t, K.T)
    return (pts2d3.T/pts2d3[:, 2]).T[:, 0:2].reshape((n_pts, n_depths, 2)).astype(int)


def GetPatch(pts2d, img, patch_size):
    """
    Get patch by coordinates

    Parameters
    ----------
    pts2d : coordinates, shape (n, 2), n is number of points, format y, x
    img : patch comes from which image
    patch_size: size of the patch
    """
    w = (patch_size - 1) // 2
    patchidx_kernel = np.array(list(product(range(-w, w+1), range(-w, w+1)))).reshape((patch_size, patch_size, 2))
    patchidx = np.stack([pts2d] * patch_size ** 2).reshape((patch_size, patch_size, *pts2d.shape)).transpose(2, 0, 1, 3)
    patchidx += patchidx_kernel
    pts2dmask = np.logical_and(0 <= patchidx, patchidx < img.shape[:2]).reshape(patchidx.shape[0], -1).all(axis=1)
    patchidx_masked = patchidx[pts2dmask, ...]
    y, x = patchidx_masked.reshape(-1, 2).T
    patch = np.zeros((*patchidx.shape[0:3], *img.shape[2:])).astype(int)
    patch[pts2dmask, ...] = img[y, x, ...].reshape((*patchidx_masked.shape[0:3], *img.shape[2:]))
    return patch


def ComputeConsistency(patch0, patch1):
    patch0 = patch0.reshape(patch0.shape[0], -1)
    patch1 = patch1.reshape(*patch1.shape[0:2], -1)
    patch0_mean, patch0_std = patch0.mean(axis=1), patch0.std(axis=1)
    patch1_mean, patch1_std = patch1.mean(axis=2), patch1.std(axis=2)
    mean = np.mean((patch0.T - patch0_mean).T * (patch1.transpose(2, 0, 1) - patch1_mean).transpose(2, 1, 0), axis=2)
    std = patch0_std * patch1_std.T
    std[std < 1e-6] = 1e-6
    corr = (mean / std).T
    return corr


def get_depth(img, extrinsic, imgs, extrinsics, patch_size, depths):
    """
    creates a depth map from a disparity map (DISPM).
    """
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 40
    pts2d0 = np.array(np.where(mask)).T
    pts3d = Get3dCoord(pts2d0, extrinsic, depths)
    patch0 = GetPatch(pts2d0.reshape(-1, 2), img, patch_size)
    corr_total = np.zeros((patch0.shape[0], len(depths)))
    for e, im in zip(extrinsics, imgs):
        pts2d1 = GetProjCoord(pts3d, e)
        patch1 = GetPatch(pts2d1.reshape(-1, 2), im, patch_size)
        patch1 = patch1.reshape(*pts2d1.shape[0:2], *patch1.shape[1:])
        corr = ComputeConsistency(patch0, patch1)
        corr_total += corr
    depths_idx = np.argmax(corr_total, axis=1)
    pass
