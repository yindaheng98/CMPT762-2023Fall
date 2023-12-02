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


debug_SaveProjCoord_n = 0


def debug_SaveProjCoord(pts2d, img):
    import os
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(0, pts2d.shape[0], 100):
        y, x = pts2d[i, [0, -1], :].T
        plt.plot(x, y, linewidth=1)
    global debug_SaveProjCoord_n
    debug_SaveProjCoord_n += 1
    plt.axis('image')
    os.makedirs(f"../results/debug", exist_ok=True)
    plt.savefig(f'../results/debug/{debug_SaveProjCoord_n}.png', dpi=300)
    plt.close()


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
    patchidx_kernel = np.array(list(product(range(-w, w+1), range(-w, w+1))))
    patchidx = np.stack([pts2d] * patch_size ** 2, axis=1)
    patchidx += patchidx_kernel
    pts2dmask = np.logical_and(0 <= patchidx, patchidx < img.shape[:2]).all(axis=2)
    patchidx_masked = patchidx[pts2dmask, ...]
    y, x = patchidx_masked.T
    patch = np.zeros((*patchidx.shape[0:2], *img.shape[2:])).astype(int)
    patch[pts2dmask, ...] = img[y, x, ...]
    return patch


debug_SavePatch_n = 0


def debug_SavePatch(patch0, patch1):  # for debug
    import random
    import os
    n, s = patch0.shape[:2]
    patch_size = int(np.sqrt(s))
    patch_shape = (patch_size, patch_size, patch0.shape[2])
    k = random.randint(0, n)
    global debug_SavePatch_n
    debug_SavePatch_n += 1
    os.makedirs(f"../results/debug/patch{debug_SavePatch_n}", exist_ok=True)
    cv2.imwrite(f"../results/debug/patch{debug_SavePatch_n}/0.png", patch0[k, ...].reshape(patch_shape))
    for i in range(patch1.shape[1]):
        cv2.imwrite(f"../results/debug/patch{debug_SavePatch_n}/{i+1}.png", patch1[k, i, ...].reshape(patch_shape))


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


def get_depth(img, extrinsic, mask, imgs, extrinsics, patch_size, depths):
    """
    creates a depth map from a disparity map (DISPM).
    """
    pts2d0 = np.array(np.where(mask)).T
    pts3d = Get3dCoord(pts2d0, extrinsic, depths)
    patch0 = GetPatch(pts2d0.reshape(-1, 2), img, patch_size)
    n, corr_total = 0, np.zeros((patch0.shape[0], len(depths)))
    for e, im in zip(extrinsics, imgs):
        pts2d1 = GetProjCoord(pts3d, e)
        debug_SaveProjCoord(pts2d1, im)
        patch1 = GetPatch(pts2d1.reshape(-1, 2), im, patch_size)
        patch1 = patch1.reshape(*pts2d1.shape[0:2], *patch1.shape[1:])
        debug_SavePatch(patch0, patch1)
        corr = ComputeConsistency(patch0, patch1)
        corr_total += corr
        n += 1
    corr_total /= n
    depths_idx = np.argmax(corr_total, axis=1)
    depths_mask = np.max(corr_total, axis=1) > 1e-6
    depthsidxmap, depthsmap = np.zeros(img.shape[:2]).astype(int), np.zeros(img.shape[:2])
    y, x = pts2d0[depths_mask, ...].T
    depthsidxmap[y, x] = depths_idx[depths_mask]
    depthsmap[y, x] = depths[depths_idx[depths_mask]]
    K, R, t = extrinsic
    pts2dxyz = np.concatenate([pts2d0[depths_mask], np.expand_dims(depths[depths_idx[depths_mask]], axis=1)], axis=1)
    P = np.dot(K, np.concatenate((R.T, [t])).T)
    pts3d = np.dot(pts2dxyz - P[:, 3], np.linalg.inv(P[:, 0:3]).T)
    colors = img[y, x, ...]
    return pts3d, colors, depthsmap, depthsidxmap
