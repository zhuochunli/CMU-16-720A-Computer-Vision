import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    heights = [bbox[2] - bbox[0] for bbox in bboxes]
    mean_height = sum(heights) / len(heights)   # find the average height

    # Each row of the matrix should contain [y1,x1,y2,x2]
    centers = [((bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2, bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes]
    centers = sorted(centers, key=lambda x: x[0])   # sort based on y1
    rows = []
    y_prev = centers[0][0]

    cur_row = []
    for center in centers:
        if center[0] > y_prev + mean_height:    # find a new row
            cur_row = sorted(cur_row, key=lambda x: x[1])
            rows.append(cur_row)
            cur_row = [center]
            y_prev = center[0]
        else:
            cur_row.append(center)
    cur_row = sorted(cur_row, key=lambda x: x[1])
    rows.append(cur_row)


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    dataset = []
    for row in rows:
        cur_data = []
        for cor_y, cor_x, h, w in row:
            # crop out the character
            img_crop = bw[cor_y - h // 2:cor_y + h // 2, cor_x - w // 2:cor_x + w // 2]
            if h > w:
                h_pad = h // 20
                w_pad = (h - w) // 2 + h_pad
            else:
                w_pad = w // 20
                h_pad = (w - h) // 2 + w_pad

            img_crop = np.pad(img_crop, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=(1, 1))
            img_crop = skimage.transform.resize(img_crop, (32, 32))     # resize image to 32*32
            img_crop = skimage.morphology.erosion(img_crop)
            img_crop = np.transpose(img_crop)   # before you flatten, transpose the image
            cur_data.append(img_crop.flatten())
        dataset.append(np.array(cur_data))
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    for data in dataset:
        h1 = forward(data, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        pred_index = np.argmax(probs, axis=1)
        pred_text = ''
        for i in pred_index:
            pred_text += letters[i]
        print(pred_text)
    