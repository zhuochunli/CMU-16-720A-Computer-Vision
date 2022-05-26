import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from scipy import ndimage
from matplotlib import pyplot as plt
# Q2.1.6


def rotTest(opts):

    # Read the image and convert to grayscale, if necessary
    im = cv2.imread('../data/cv_cover.jpg')
    his = []
    for i in range(36):
        # Rotate Image
        im_r = ndimage.rotate(im, 10*i)
        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(im, im_r, opts)
        # Update histogram
        his.append(matches.shape[0])

        pass

    # Display histogram
    degrees = [i*10 for i in range(36)]
    plt.bar(degrees, his, width=9)
    plt.show()


if __name__ == "__main__":
    opts = get_opts()
    rotTest(opts)
