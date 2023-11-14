import numpy as np
from scipy.optimize import minimize

def refineF(F, pts1, pts2):
    
    X = np.column_stack((pts1[:, 0], pts2[:, 0]))
    Y = np.column_stack((pts1[:, 1], pts2[:, 1]))
    initF = F.copy()

    # Do the minimization
    res = minimize(svd_distance, initF, args=(X, Y), method='L-BFGS-B', options={'maxiter': 100000, 'maxfun': 10000})
    minF = res.x.reshape((3, 3))
    F = rank2F(minF)
    return F

def svd_distance(F, X, Y):
    F = F.reshape((3, 3))
    F = rank2F(F)

    homogPoints = np.column_stack((X[:, 0], Y[:, 0], np.ones(X.shape[0])))
    homogPointsp = np.column_stack((X[:, 1], Y[:, 1], np.ones(X.shape[0])))

    FX = np.dot(F, homogPoints.T)
    FTXp = np.dot(F.T, homogPointsp.T)

    dist = np.array([((homogPointsp[i, :] @ FX[:, i]) ** 2) * ((1 / (FX[0, i] ** 2 + FX[1, i] ** 2)) + (1 / (FTXp[0, i] ** 2 + FTXp[1, i] ** 2))) for i in range(X.shape[0])])
    d = np.sum(dist)
    return d

def rank2F(F):
    U, W, Vt = np.linalg.svd(F)
    W[2] = 0
    F2 = U @ np.diag(W) @ Vt
    return F2
