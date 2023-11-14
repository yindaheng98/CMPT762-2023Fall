import numpy as np
import matplotlib.pyplot as plt
import cv2
from get_depth import get_depth
from get_disparity import get_disparity

# Load image and parameters
im1 = cv2.imread('../data/im1.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../data/im2.png', cv2.IMREAD_GRAYSCALE)

# Load rectify.mat parameters
rectify_params = np.load('../results/rectify.npy', allow_pickle=True).tolist()
M1 = rectify_params['M1']
M2 = rectify_params['M2']
K1n = rectify_params['K1n']
K2n = rectify_params['K2n']
R1n = rectify_params['R1n']
R2n = rectify_params['R2n']
t1n = rectify_params['t1n']
t2n = rectify_params['t2n']

maxDisp = 20
windowSize = 3
# Assuming get_disparity is a predefined function
dispM = get_disparity(im1, im2, maxDisp, windowSize)

# --------------------  get depth map
# Assuming get_depth is a predefined function
depthM = get_depth(dispM, K1n, K2n, R1n, R2n, t1n, t2n)

# --------------------  Display
plt.figure()
plt.imshow(dispM * (im1 > 40), cmap='gray')
plt.axis('image')
plt.title('Disparity Map')
plt.savefig('../results/disparity.png', dpi=300)
plt.close()

plt.figure()
plt.imshow(depthM * (im1 > 40), cmap='gray')
plt.axis('image')
plt.title('Depth Map')
plt.savefig('../results/depth.png', dpi=300)
