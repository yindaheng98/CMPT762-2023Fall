import cv2
import numpy as np

def warp_stereo(IL, IR, TL, TR):
    bb = mcbb(IL.shape, IR.shape, TL, TR)

    if bb[2] - bb[0] > 3000 or bb[3] - bb[1] > 3000:
        raise ValueError("XX Error: Your rectification is not correct. Debug it before going further...")

    JL, bbL = imwarp(IL, TL, 'bilinear', bb)
    JR, bbR = imwarp(IR, TR, 'bilinear', bb)
    
    return JL, JR, bbL, bbR

def mcbb(s1, s2, H1, H2):
    corners = np.array([[0, 0, s1[1], s1[1]], [0, s1[0], 0, s1[0]]])
    corners_x = p2t(H1, corners)

    minx = np.floor(min(corners_x[0, :])).astype(np.int32)
    maxx = np.ceil(max(corners_x[0, :])).astype(np.int32)
    miny = np.floor(min(corners_x[1, :])).astype(np.int32)
    maxy = np.ceil(max(corners_x[1, :])).astype(np.int32)
    bb1 = [minx, miny, maxx, maxy]

    corners = np.array([[0, 0, s2[1], s2[1]], [0, s2[0], 0, s2[0]]])
    corners_x = p2t(H2, corners)

    minx = np.floor(min(corners_x[0, :])).astype(np.int32)
    maxx = np.ceil(max(corners_x[0, :])).astype(np.int32)
    miny = np.floor(min(corners_x[1, :])).astype(np.int32)
    maxy = np.ceil(max(corners_x[1, :])).astype(np.int32)
    bb2 = [minx, miny, maxx, maxy]

    q1 = np.min(np.array([bb1, bb2]), axis=0)
    q2 = np.max(np.array([bb1, bb2]), axis=0)

    bb = [q1[0], q1[1], q2[2], q2[3]]

    return bb

def imwarp(I, H, meth='linear', sz='same'):
    if H.shape != (3, 3):
        raise ValueError('Invalid input transformation')

    if sz == 'same':
        minx = 0
        maxx = I.shape[1]
        miny = 0
        maxy = I.shape[0]
    elif len(sz) == 4 and isinstance(sz, list):
        # force the bounding box
        minx = sz[0]
        miny = sz[1]
        maxx = sz[2]
        maxy = sz[3]
    else:
        raise ValueError('Invalid size parameter')

    bb = [minx, miny, maxx, maxy]

    x, y = np.meshgrid(range(minx, maxx), range(miny, maxy))
    pp = p2t(np.linalg.inv(H), np.vstack((x.flatten(), y.flatten())))

    xi = pp[0, :].reshape(x.shape)
    yi = pp[1, :].reshape(y.shape)

    I2 = cv2.remap(I, xi.astype(np.float32), yi.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return I2, bb


def p2t(H, m):
    """
    Apply a projective (homographic) transformation in 2D.
    
    Parameters:
        H (numpy.ndarray): a 3x3 homography matrix.
        m (numpy.ndarray): a set of image points.
        
    Returns:
        numpy.ndarray: the transformed points.
    """
    
    # Check the shape of the transformation matrix
    if H.shape != (3, 3):
        raise ValueError('Invalid format of the transformation matrix (3x3)!')
    
    # Check the shape of the image points
    if m.shape[0] != 2:
        raise ValueError('Image coordinates must be Cartesian!')
    
    # Applying the homography to the points
    num_points = m.shape[1]
    homogeneous_coords = np.vstack((m, np.ones((1, num_points))))
    transformed_coords = np.dot(H, homogeneous_coords)
    transformed_coords = transformed_coords / transformed_coords[2, :]
    
    return transformed_coords[0:2, :]

