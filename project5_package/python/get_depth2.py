import numpy as np
import cv2
from itertools import product


def Get3dCoord(pts2d, extrinsic, depths):
    n_pts, n_depths = pts2d.shape[0], depths.shape[0]
    K, R, t = extrinsic
    P = np.dot(K, np.concatenate((R.T, [t])).T)
    dpts2d = np.stack([pts2d] * depths.shape[0], axis=1) * np.stack([depths] * pts2d.shape[1], axis=1)
    dpts2d3 = np.concatenate([dpts2d, np.expand_dims(np.zeros(dpts2d.shape[0:2]) + depths, axis=2)], axis=2)
    pts3d = np.dot(
        dpts2d3.reshape((n_pts * n_depths, 3)) - P[:, 3],
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
    # all provided image contains all the region, so do not need a mask
    # ptsmapmask = (pts2dmap.reshape(-1, 2) < img.shape[:2]).reshape(pts2dmap.shape)
    w = (patch_size - 1) // 2
    patchidx_kernel = np.array(list(product(range(-w, w+1), range(-w, w+1)))).reshape((patch_size, patch_size, 2))
    patchidx = np.stack([pts2d] * patch_size ** 2).reshape((patch_size, patch_size, *pts2d.shape)).transpose(2, 0, 1, 3)
    patchidx += patchidx_kernel
    y, x = patchidx.reshape(-1, 2).T
    patch = img[y, x, ...].reshape((*patchidx.shape[0:3], -1))
    return patch


def get_depth(img, extrinsic, imgs, extrinsics, patch_size, depths):
    """
    creates a depth map from a disparity map (DISPM).
    """
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 40
    pts2d0 = np.array(np.where(mask)).T
    pts3d = Get3dCoord(pts2d0, extrinsic, depths)
    patch0 = GetPatch(pts2d0.reshape(-1, 2), img, patch_size)
    for e, im in zip(extrinsics, imgs):
        pts2d1 = GetProjCoord(pts3d, e)
        patch1 = GetPatch(pts2d1.reshape(-1, 2), im, patch_size)
        patch1 = patch1.reshape(*pts2d1.shape[0:2], *patch1.shape[1:])
        pass
    pass
