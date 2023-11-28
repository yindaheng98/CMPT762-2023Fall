import numpy as np
from numpy.linalg import svd


def triangulate(P1, pts1, P2, pts2):
    """
    Estimate the 3D positions of points from 2d correspondence
    Args:
        P1:     projection matrix with shape 3 x 4 for image 1
        pts1:   coordinates of points with shape N x 2 on image 1
        P2:     projection matrix with shape 3 x 4 for image 2
        pts2:   coordinates of points with shape N x 2 on image 2

    Returns:
        Pts3d:  coordinates of 3D points with shape N x 3
    """
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))
    for i in range(num_points):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        A = np.stack([
            P1[0].T-u1*P1[2].T,
            P1[1].T-v1*P1[2].T,
            P2[0].T-u2*P2[2].T,
            P2[1].T-v2*P2[2].T,
        ])
        SVDResults = svd(A)
        U, S, V = SVDResults.U, SVDResults.S, SVDResults.Vh
        v = V[np.argmin(S)]
        x = v/v[-1]
        pts3d[i, :] = x[0:3]

    return pts3d
