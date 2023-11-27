import numpy as np
from numpy.linalg import svd
from numpy.linalg import matrix_rank
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    
    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'
    # pts1_xyz = np.concatenate([pts1, np.zeros((pts1.shape[0], 1)) + M], axis=1) # same as divide by M
    # pts2_xyz = np.concatenate([pts2, np.zeros((pts2.shape[0], 1)) + M], axis=1) # same as divide by M
    pts1_xyz = np.concatenate([pts1 / M, np.ones((pts1.shape[0], 1))], axis=1)
    pts2_xyz = np.concatenate([pts2 / M, np.ones((pts2.shape[0], 1))], axis=1)
    A = np.reshape(np.stack([pts1_xyz] * 3, axis=2) * np.stack([pts2_xyz] * 3, axis=1), (-1, 9))
    SVDResults = svd(A)
    U, S, V = SVDResults.U, SVDResults.S, SVDResults.Vh
    v = V[np.argmin(S)]
    F3 = v.reshape((3, 3))
    SVDResults = svd(F3)
    U, S, V = SVDResults.U, SVDResults.S, SVDResults.Vh
    S[-1] = 0
    F = np.dot(U * S.reshape((1, -1)), V)
    return F
