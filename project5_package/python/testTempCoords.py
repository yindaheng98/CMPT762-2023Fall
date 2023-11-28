import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI

# Load images and points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']
M = pts['M']

# write your code here
F = eightpoint(pts1, pts2, M)
# displayEpipolarF(img1, img2, F)
# epipolarMatchGUI(img1, img2, F)
intrinsics = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
K1, K2 = intrinsics['K1'], intrinsics['K2']
E = essentialMatrix(F, K1, K2)
M2s = camera2(E)
P1 = np.concatenate([np.diag([1, 1, 1]), np.zeros((3, 1))], axis=1)
P2, pts3d = None, None
for i in range(M2s.shape[2]):
    P2_ = M2s[:, :, i]
    pts3d_ = triangulate(np.dot(K1, P1), pts1, np.dot(K2, P2_), pts2)
    pts3d1 = np.concatenate([pts3d_, np.ones((pts3d_.shape[0], 1))], axis=1)
    pts1_c, pts2_c = np.dot(pts3d1, P1.T), np.dot(pts3d1, P2_.T)
    z1_min, z2_min = min(pts1_c[:, -1]), min(pts2_c[:, -1])
    if z1_min >= 0 and z2_min >= 0:
        P2 = P2_
        pts3d = pts3d_
R1, t1 = P1[:, 0:3], P1[:, 3]
R2, t2 = P2[:, 0:3], P2[:, 3]

# save extrinsic parameters for dense reconstruction
np.save('../results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})

pts3d1 = np.concatenate([pts3d, np.ones((pts3d.shape[0], 1))], axis=1)
pts1_, pts2_ = np.dot(pts3d1, np.dot(K1, P1).T), np.dot(pts3d1, np.dot(K2, P2).T)
pts1_, pts2_ = (pts1_/pts1_[:, -1:])[:, 0:2], (pts2_/pts2_[:, -1:])[:, 0:2]
print(np.average(np.abs(pts1_-pts1)), np.average(np.abs(pts2_-pts2)))
