import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.interpolate import RectBivariateSpline

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)

    image1_warp = affine_transform(image1, M, image1.shape)
    diff = np.abs(image2 - image1_warp)
    mask[diff < tolerance] = 0

    mask = binary_dilation(mask, np.ones((3, 3)))
    mask = binary_erosion(mask, np.ones((3, 3)))
    return mask