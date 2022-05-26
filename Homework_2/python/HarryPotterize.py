import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
import matchPics
import planarH
import matplotlib.pyplot as plt
# Import necessary functions

# Q2.2.4

def warpImage(opts):
    opts = get_opts()
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    matches, locs1, locs2 = matchPics.matchPics(cv_cover, cv_desk, opts)

    # flip the RGB channel to print original picture in the result
    hp_cover = np.flip(hp_cover, axis=2)
    cv_desk = np.flip(cv_desk, axis=2)

    # scaling
    x1, x2 = np.zeros(locs1.shape), np.zeros(locs2.shape)
    x1[:, 0] = locs1[:, 1] * hp_cover.shape[1] / cv_cover.shape[1]
    x1[:, 1] = locs1[:, 0] * hp_cover.shape[0] / cv_cover.shape[0]
    x2[:, 0], x2[:, 1] = locs2[:, 1], locs2[:, 0]
    locs1, locs2 = x1[matches[:, 0]], x2[matches[:, 1]]

    # Use matching locs1 and locs2 to compute H
    bestH2to1, inliers = planarH.computeH_ransac(locs1, locs2, opts)

    # Use H to transform hp cover, with same output size as cv desk
    composite = planarH.compositeH(bestH2to1, hp_cover, cv_desk)
    plt.imshow(composite)
    # hide x-axis and y-axis
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()


if __name__ == "__main__":
    opts = get_opts()
    warpImage(opts)


