import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    computes the pose matrix (camera matrix) P given 2D and 3D
    points.
    
    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    """
    X = np.concatenate([X, np.ones((1, *X.shape[1:]))])
    Al = np.stack([np.concatenate([X, np.zeros_like(X)]), np.concatenate([np.zeros_like(X), X])], axis=1)
    Ar = -x * np.stack([X] * x.shape[0], axis=1)
    A = np.concatenate([Al, Ar]).reshape(((X.shape[0] + x.shape[0]) * 2, x.shape[0] * X.shape[1], *X.shape[2:]), order='F').T
    U, s, Vh = svd(A)
    v = Vh[np.argmin(s)]
    P = v.reshape((3, 4))
    return P
