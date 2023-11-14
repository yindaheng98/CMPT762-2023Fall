import numpy as np
import matplotlib.pyplot as plt

def displayEpipolarF(I1, I2, F):
    e1, e2 = epipoles(F)

    sy, sx = I2.shape[:2]

    plt.figure()
    plt.subplot(121)
    plt.imshow(I1[..., ::-1])
    plt.xlabel('Select a point in this image Right-click when finished')

    plt.subplot(122)
    plt.imshow(I2[..., ::-1])
    plt.xlabel('Verify that the corresponding point is on the epipolar line in this image')

    while True:
        plt.subplot(121)
        plt.legend(['show'])
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

        if 0 < x < I1.shape[1] and 0 < y < I1.shape[0]:
            pass
        else:
            plt.subplot(121)
            plt.title('Epipole is outside image boundary')

        plt.subplot(121)
        
        xc = x
        yc = y

        v = np.array([xc, yc, 1])

        l = np.dot(F, v.T)

        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            raise Exception('Zero line vector in displayEpipolar')

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

        plt.subplot(121)
        plt.plot(x, y, '*', markersize=6, linewidth=2)
        
        plt.subplot(122)
        plt.xlim(0, I2.shape[1])
        plt.ylim(I2.shape[0], 0)
        plt.plot([xs, xe], [ys, ye])
        plt.draw()


def epipoles(E):
    U, S, V = np.linalg.svd(E)

    e1 = V.T[:, -1]

    U, S, V = np.linalg.svd(E.T)

    e2 = V.T[:, -1]

    return e1, e2