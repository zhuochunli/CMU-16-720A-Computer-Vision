import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches
from opts import get_opts
# Q2.1.4


def matchPics(I1, I2, opts):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    matches, locs1, locs2 = None, None, None

    # TODO: Convert Images to GrayScale
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # TODO: Detect Features in Both Images
    locs1 = corner_detection(I1, sigma)
    locs2 = corner_detection(I2, sigma)

    # TODO: Obtain descriptors for the computed feature locations
    I1_desc, locs1 = computeBrief(I1, locs1)
    I2_desc, locs2 = computeBrief(I2, locs2)

    # TODO: Match features using the descriptors
    matches = briefMatch(I1_desc, I2_desc, ratio)

    return matches, locs1, locs2


if __name__ == "__main__":
    opts = get_opts()
    im1 = cv2.imread('../data/cv_cover.jpg')
    im2 = cv2.imread('../data/cv_desk.png')
    matches, locs1, locs2 = matchPics(im1, im2, opts)
    plotMatches(im1, im2, matches, locs1, locs2)