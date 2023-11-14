import cv2
import numpy as np
import matplotlib.pyplot as plt
from rectify_pair import rectify_pair
from warp_stereo import warp_stereo, p2t

# Load images
im1 = cv2.imread('../data/im1.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../data/im2.png', cv2.IMREAD_GRAYSCALE)

# Load intrinsic and extrinsic parameters
K = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
K1, K2 = K['K1'], K['K2']
P = np.load('../results/extrinsics.npy', allow_pickle=True).tolist()
R1, R2 = P['R1'], P['R2']
t1, t2 = P['t1'], P['t2']

M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = rectify_pair(K1, K2, R1, R2, t1, t2)

rectIL, rectIR, bbL, bbR = warp_stereo(im1, im2, M1, M2)

# save the rectification parameters
np.save('../results/rectify', {'M1': M1, 'M2': M2, 'K1n': K1n, 'K2n': K2n, 
                               'R1n':R1n, 'R2n':R2n, 't1n':t1n, 't2n':t2n})

# Display rectified images
rectified_images = np.hstack((rectIL, rectIR))
plt.imshow(rectified_images, cmap='gray')
plt.title('Rectified Stereo Images')
plt.show()

# Load ground truth points
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1, pts2 = pts['pts1'], pts['pts2']

# Warp left and right points
gtL = pts1[::20].T
gtR = pts2[::20].T
mlx = p2t(M1, gtL)
mrx = p2t(M2, gtR)
mrx[0, :] += rectIL.shape[1]

# Display points on rectified images
plt.imshow(rectified_images, cmap='gray')
plt.plot(mlx[0, :] - bbL[0], mlx[1, :] - bbL[1], 'r*', markersize=10)
plt.plot(mrx[0, :] - bbR[0], mrx[1, :] - bbR[1], 'b*', markersize=10)
# Draw horizontal lines connecting matched points
for i in range(mlx.shape[1]):
    plt.plot([0, rectified_images.shape[1]-1],
             [mlx[1, i] - bbL[1], mrx[1, i] - bbR[1]], 'g')
plt.title('Rectified Stereo Images with Ground Truth Points')
plt.show()
