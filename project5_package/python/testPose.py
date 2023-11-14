import numpy as np
from scipy.linalg import svd
from estimate_pose import estimate_pose


# Random generate camera matrix
K = np.array([[1, 0, 1e2], [0, 1, 1e2], [0, 0, 1]])

# Random generate R and t
R, _, _ = svd(np.random.randn(3, 3))
t = np.random.randn(3, 1)

P = K @ np.hstack([R, t])

# Random generate 2D and 3D points
N = 10
X = np.random.randn(3, N)
x = P @ np.vstack([X, np.ones((1, N))])
x[0, :] /= x[2, :]
x[1, :] /= x[2, :]
x = x[:2, :]

# Test
PClean = estimate_pose(x, X)

xProj = PClean @ np.vstack([X, np.ones((1, N))])
xProj[0, :] /= xProj[2, :]
xProj[1, :] /= xProj[2, :]
xProj = xProj[:2, :]

print(f'Reprojected Error with clean 2D points is {np.linalg.norm(xProj - x):.4f}')
print(f'Pose Error with clean 2D points is {np.linalg.norm(PClean/PClean.reshape(-1)[-1] - P/P.reshape(-1)[-1]):.4f}')

# Noise performance
# Add some noise
xNoise = x + np.random.rand(*x.shape)

PNoisy = estimate_pose(xNoise, X)

xProj = PNoisy @ np.vstack([X, np.ones((1, N))])
xProj[0, :] /= xProj[2, :]
xProj[1, :] /= xProj[2, :]
xProj = xProj[:2, :]

print('------------------------------')
print(f'Reprojected Error with noisy 2D points is {np.linalg.norm(xProj - x):.4f}')
print(f'Pose Error with noisy 2D points is {np.linalg.norm(PNoisy/PNoisy.reshape(-1)[-1] - P/P.reshape(-1)[-1])/np.linalg.norm(P/P.reshape(-1)[-1]):.4f}')

