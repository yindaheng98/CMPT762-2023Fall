from itertools import product
import numpy as np
import cv2
import matplotlib.pyplot as plt
from get_depth2 import GetProjCoord, get_corr

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

minx, miny, minz = -0.023121, -0.038009, -0.091940
maxx, maxy, maxz = 0.078626, 0.121636, -0.017395
pts3d = np.array([(x, y, z) for x, y, z in product((minx, maxx), (miny, maxy), (minz, maxz))])
for i in range(len(img_names)):
    name = img_names[i]
    pts2d = GetProjCoord(pts3d, KRts[name])
    img = cv2.imread("../data/" + name)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(x=pts2d[:, 0], y=pts2d[:, 1], marker='x')
    plt.axis('image')
    plt.savefig(f'../results/corners{i}.png', dpi=300)
    plt.savefig(f'../results/corners{i}.pdf', dpi=300)
    plt.close()

camera_pose = -np.dot(t, np.linalg.inv(R).T)
distance = np.linalg.norm(pts3d - camera_pose, axis=1)
depths = np.linspace(np.min(distance), np.max(distance), 256)
patch_size = 5
img = cv2.imread("../data/" + img_names[0])
mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 32
pts2d0 = np.array(np.where(mask)).T[:, ::-1]
debug_img = np.zeros_like(img)
debug_img[mask, :] = img[mask, :]
cv2.imwrite('../results/debug/maskedimg.png', debug_img)
corr = get_corr(
    img,
    KRts[img_names[0]],
    pts2d0,
    [cv2.imread("../data/" + name) for name in img_names[1:]],
    [KRts[name] for name in img_names[1:]],
    patch_size, depths
)
np.savez("../results/pts2dcorr.npz", pts2d0=pts2d0, corr=corr, depths=depths)
