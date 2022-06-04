import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    #denoise
    img = skimage.restoration.denoise_bilateral(image, multichannel=True)
    # greyscale
    img_grey = skimage.color.rgb2gray(img)

    # threshold, morphology
    thresh = skimage.filters.threshold_otsu(img_grey)
    bw = img_grey < thresh
    bw = skimage.morphology.closing(bw, skimage.morphology.square(5))

    # label
    label_img = skimage.morphology.label(bw, connectivity=2)
    properties = skimage.measure.regionprops(label_img)

    # skip small boxes
    mean_area = sum([x.area for x in properties]) / len(properties)
    for x in properties:
        if x.area > mean_area/4:
            bboxes.append(x.bbox)

    bw = (~bw).astype(float)

    return bboxes, bw
