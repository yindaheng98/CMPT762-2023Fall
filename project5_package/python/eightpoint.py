import numpy as np
from numpy.linalg import svd
from numpy.linalg import matrix_rank
from refineF import refineF, rank2F

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """

    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'

    # pts1_xyz = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1) # what we really should do, but unstable?
    # pts2_xyz = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1) # what we really should do, but unstable?
    T1 = np.array([
        [1/np.std(pts1[:, 0]), 0, -np.mean(pts1[:, 0])/np.std(pts1[:, 0])],
        [0, 1/np.std(pts1[:, 1]), -np.mean(pts1[:, 1])/np.std(pts1[:, 1])],
        [0, 0, 1]
    ])
    T2 = np.array([
        [1/np.std(pts2[:, 0]), 0, -np.mean(pts2[:, 0])/np.std(pts2[:, 0])],
        [0, 1/np.std(pts2[:, 1]), -np.mean(pts2[:, 1])/np.std(pts2[:, 1])],
        [0, 0, 1]
    ])
    pts1_xyz = np.dot(np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1), T1.T)
    pts2_xyz = np.dot(np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1), T2.T)
    A = np.reshape(np.stack([pts1_xyz] * 3, axis=2) * np.stack([pts2_xyz] * 3, axis=1), (-1, 9))
    SVDResults = svd(A)
    U, S, V = SVDResults.U, SVDResults.S, SVDResults.Vh
    v = V[np.argmin(S)]
    F = v.reshape((3, 3))
    return refineF((T1.T @ F @ T2).reshape(-1), pts1, pts2).reshape((3, 3))
    # return T1.T @ rank2F(F) @ T2
