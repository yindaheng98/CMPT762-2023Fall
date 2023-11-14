import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from estimate_params import estimate_params
from estimate_pose import estimate_pose

data = np.load("../data/PnP.npy", allow_pickle=True).tolist()
