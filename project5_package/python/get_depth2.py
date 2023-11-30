import numpy as np
import cv2

def Get3dCoord(pts2d, extrinsic, depths):
    n_pts, n_depths = pts2d.shape[0], depths.shape[0]
    K, R, t = extrinsic
    P = np.dot(K, np.concatenate((R.T, [t])).T)
    dpts2d = np.stack([pts2d] * depths.shape[0], axis=1) * np.stack([depths] * pts2d.shape[1], axis=1)
    dpts2d3 = np.concatenate([dpts2d, np.expand_dims(np.zeros(dpts2d.shape[0:2]) + depths, axis=2)], axis=2)
    pts3d = np.dot(dpts2d3.reshape((n_pts * n_depths, 3)) - P[:, 3], np.linalg.inv(P[:, 0:3]).T).reshape((n_pts, n_depths, 3))
    return pts3d
    


def get_depth(img, extrinsic, imgs, extrinsics, patch_size, depths):
    """
    creates a depth map from a disparity map (DISPM).
    """
    y_max, x_max = img.shape[:2]
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 40
    pts2d = np.array(np.where(mask)).T
    pts3d = Get3dCoord(pts2d, extrinsic, depths)
        
