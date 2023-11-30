import numpy as np
from scipy.linalg import svd, qr

def estimate_params(P):
    """
    computes the intrinsic K, rotation R, and translation t from
    given camera matrix P.
    
    Args:
        P: Camera matrix
    """
    H, h = P[:, 0:3], P[:, 3]
    q, r = qr(np.linalg.inv(H))
    R, K = q.T, np.linalg.inv(r)
    K = K/K[2, 2]
    R_pi = np.diag([K[i, i]/K[i, i] for i in range(K.shape[0])])
    K = np.dot(K, R_pi)
    R = np.dot(R_pi, R)
    K[0, 1] = 0
    t = np.dot(h, np.linalg.inv(K).T)
    return K, R, t

