import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    c1, c2 = -np.dot(t1, R1), -np.dot(t2, R2)
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r2 = np.cross(R1[:, 2], r1)
    r3 = np.cross(r1, r2)
    R1n = R2n = np.array([r1, r2, r3]).T
    K1n = K2n = K1
    t1n, t2n = -np.dot(c1, R1n), -np.dot(c2, R2n)
    M1, M2 = np.dot(np.dot(K1n, R1n), np.linalg.inv(np.dot(K1, R1))), np.dot(np.dot(K2n, R2n), np.linalg.inv(np.dot(K2, R2)))

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

