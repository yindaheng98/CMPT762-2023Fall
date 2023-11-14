import numpy as np
from scipy.linalg import svd
from estimate_pose import estimate_pose
from estimate_params import estimate_params


# Randomly generate camera matrix
K = np.array([[1, 0, 1e2], [0, 1, 1e2], [0, 0, 1]])

R, _, _ = svd(np.random.randn(3, 3))

if np.linalg.det(R) < 0:
    R = -R
t = np.random.randn(3, 1)

P = K @ np.hstack([R, t])

# Randomly generate 2D and 3D points
N = 10
X = np.random.randn(3, N)
x = P @ np.vstack([X, np.ones((1, N))])
x[0, :] /= x[2, :]
x[1, :] /= x[2, :]
x = x[:2, :]

# Test
PClean = estimate_pose(x, X)
KClean, RClean, tClean = estimate_params(PClean)
print(f'Intrinsic Error with clean 2D points is {np.linalg.norm(KClean/KClean.reshape(-1)[-1] - K/K.reshape(-1)[-1]):.4f}')
print(f'Rotation Error with clean 2D points is {np.linalg.norm(RClean - R):.4f}')
print(f'Translation Error with clean 2D points is {np.linalg.norm(tClean - t):.4f}')

# Noise performance - add some noise
xNoise = x + np.random.rand(*x.shape)
PNoisy = estimate_pose(xNoise, X)
KNoisy, RNoisy, tNoisy = estimate_params(PNoisy)
print('------------------------------')
print(f'Intrinsic Error with noisy 2D points is {np.linalg.norm(KNoisy/KNoisy.reshape(-1)[-1] - K/K.reshape(-1)[-1]):.4f}')
print(f'Rotation Error with noisy 2D points is {np.linalg.norm(RNoisy - R):.4f}')
print(f'Translation Error with noisy 2D points is {np.linalg.norm(tNoisy - t):.4f}')

