from itertools import product
import numpy as np
import cv2
import matplotlib.pyplot as plt
from get_depth2 import get_depth

extrinsics = {}
img_names = []
with open("../data/templeR_par.txt", "r") as f:
    for line in f.readlines():
        split = line.strip().split(" ")
        name = split[0]
        data = [float(s) for s in split[1:]]
        K = np.array(data[0:9]).reshape((3, 3))
        R = np.array(data[9:18]).reshape((3, 3))
        t = np.array(data[18:21])
        extrinsics[split[0]] = (K, R, t)
        img_names.append(split[0])

minx, miny, minz = -0.023121, -0.038009, -0.091940
maxx, maxy, maxz = 0.078626, 0.121636, -0.017395
pts3d = np.array([(x, y, z) for x, y, z in product((minx, maxx), (miny, maxy), (minz, maxz))])
K, R, t = extrinsics[img_names[0]]
pts2d = np.dot(np.dot(pts3d, R.T) + t, K.T)
pts2d = (pts2d[:, 0:2].T / pts2d[:, 2]).T.astype(int)
im0 = cv2.cvtColor(cv2.imread("../data/" + img_names[0]), cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(im0)
plt.scatter(x=pts2d[:, 0], y=pts2d[:, 1], marker='x')
plt.axis('image')
plt.savefig('../results/corners.png', dpi=300)
plt.close()

camera_pose = -np.dot(t, np.linalg.inv(R).T)
distance = np.linalg.norm(pts3d - camera_pose, axis=1)
depths = np.linspace(np.min(distance), np.max(distance), 16)
patch_size = 5
depth = get_depth(
    im0,
    extrinsics[img_names[0]],
    [cv2.imread("../data/" + name) for name in img_names[1:]],
    [extrinsics[name] for name in img_names[1:]],
    patch_size, depths
)
