import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from get_depth2 import get_depth

KRts = {}
img_names = []
with open("../data/templeR_par.txt", "r") as f:
    for line in f.readlines():
        split = line.strip().split(" ")
        name = split[0]
        data = [float(s) for s in split[1:]]
        K = np.array(data[0:9]).reshape((3, 3))
        R = np.array(data[9:18]).reshape((3, 3))
        t = np.array(data[18:21])
        KRts[split[0]] = (K, R, t)
        img_names.append(split[0])

img = cv2.imread("../data/" + img_names[0])

pts2dcorr = np.load("../results/pts2dcorr.npz")
pts2d0 = pts2dcorr['pts2d0']
corr = pts2dcorr['corr']
depths = pts2dcorr['depths']
corr_thr = 0.84
pts3d, colors, depthsmap, depthsidxmap = get_depth(
    img, KRts[img_names[0]], pts2d0, corr, depths, corr_thr
)

plt.figure()
plt.imshow(depthsmap, cmap='gray')
plt.axis('image')
plt.savefig('../results/depthsmap.png', dpi=300)
plt.savefig('../results/depthsmap.pdf')
plt.close()

plt.figure()
plt.imshow(depthsidxmap, cmap='gray')
plt.axis('image')
plt.savefig('../results/depthsidxmap.png', dpi=300)
plt.savefig('../results/depthsidxmap.pdf')
plt.close()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts3d)
pcd.colors = o3d.utility.Vector3dVector(colors[:, [2, 1, 0]] / 255)
o3d.io.write_point_cloud("../results/temple.pcd", pcd)
