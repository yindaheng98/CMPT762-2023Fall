import numpy as np
import cv2

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    xl, yl = pts1[0, 0], pts1[0, 1]
    l = np.dot(np.array([xl, yl, 1]), F.T)

    yr_max, xr_max = im2.shape[:2]
    if l[0] < l[1]:
        yr = np.array(list(range(yr_max)))
        xr = -(l[1] * yr + l[2]) / l[0]
    else:
        xr = np.array(list(range(xr_max)))
        yr = -(l[0] * xr + l[2]) / l[1]
    xr_, yr_ = xr, yr
    xr, yr = xr[xr>=0], yr[xr>=0]
    xr, yr = xr[yr>=0], yr[yr>=0]
    xr, yr = xr[xr<xr_max], yr[xr<xr_max]
    xr, yr = xr[yr<yr_max], yr[yr<yr_max]
    pts2 = np.array([[xr[0], yr[0]]])
    return pts2
