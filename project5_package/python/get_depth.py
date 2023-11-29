import numpy as np


def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (DISPM).
    """
    c1, c2 = -np.dot(t1, R1), -np.dot(t2, R2)
    f, b = K1[0, 0], np.linalg.norm(c1 - c2)
    dispM[dispM < 1e-6] = 1e-6
    return f * b / dispM
