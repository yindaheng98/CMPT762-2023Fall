import numpy as np
import cv2
import matplotlib.pyplot as plt
from epipolarCorrespondence import epipolarCorrespondence

def epipolarMatchGUI(I1, I2, F):
    coordsIM1 = []
    coordsIM2 = []
    sy, sx = I2.shape[:2]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(I1[..., ::-1])
    plt.xlabel('Select a point in this image\n(Right-click when finished)')

    ax2 = fig.add_subplot(122)
    ax2.imshow(I2[..., ::-1])
    plt.xlabel('Verify that the corresponding point\nis on the epipolar line in this image')

    while True:
        plt.subplot(121)

        point = plt.ginput(1, timeout=-1, mouse_stop=3)
        if len(point) == 0:
            x = y = 0
            stop = True
        else:
            x, y = point[0]
            stop = False

        plt.subplot(121)
        plt.title('')

        if stop:
            break

        xc = x
        yc = y

        v = np.array([xc, yc, 1])
        l = np.dot(F, v.T)

        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            raise ValueError('Zero line vector in displayEpipolar')

        l = l / s

        if l[0] != 0:
            ye = sy
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        spot = plt.plot(x, y, '*', markersize=10, linewidth=2)
        color = spot[0].get_color()
        
        plt.subplot(122)
        plt.xlim(0, I2.shape[1])
        plt.ylim(I2.shape[0], 0)
        plt.plot([xs, xe], [ys, ye], linewidth=2, color=color)
        pts2 = epipolarCorrespondence(I1, I2, F, np.array([[x, y]]))
        x2, y2 = pts2[:,0], pts2[:,1]
        plt.plot(x2, y2, 'o', markerfacecolor='none', markeredgewidth=3, markersize=8, color=color)
        plt.draw()
        
        coordsIM1.append([x, y])
        coordsIM2.append([x2[0], y2[0]])

    return np.array(coordsIM1), np.array(coordsIM2)
