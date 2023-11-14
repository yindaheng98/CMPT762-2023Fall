import numpy as np

def camera2(E):
    U, S, V = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2
    E = np.dot(U, np.dot(np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]]), V))

    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Make sure we return rotation matrices with det(R) == 1
    if np.linalg.det(np.dot(np.dot(U, W), V)) < 0:
        W = -W

    M2s = np.zeros((3, 4, 4))
    M2s[:, :, 0] = np.hstack((np.dot(np.dot(U, W), V), U[:, 2:] / np.max(np.abs(U[:, 2]))))
    M2s[:, :, 1] = np.hstack((np.dot(np.dot(U, W), V), -U[:, 2:] / np.max(np.abs(U[:, 2]))))
    M2s[:, :, 2] = np.hstack((np.dot(np.dot(U, W.T), V), U[:, 2:] / np.max(np.abs(U[:, 2]))))
    M2s[:, :, 3] = np.hstack((np.dot(np.dot(U, W.T), V), -U[:, 2:] / np.max(np.abs(U[:, 2]))))

    return M2s
