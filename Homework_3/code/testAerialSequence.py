import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from SubtractDominantMotion import SubtractDominantMotion
import datetime

# write your script here, we recommend the above libraries for making your animation
start = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
pre = seq[:, :, 0]

for i in range(1, seq.shape[2]):
    cur = seq[:, :, i]
    mask = SubtractDominantMotion(pre, cur, threshold, num_iters, tolerance)

    tmp = np.repeat(cur[:, :, np.newaxis], 3, 2)
    tmp[:, :, 0][mask == 1] = 1

    if i in [30, 60, 90, 120]:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(tmp, cmap='gray')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()

    pre = cur

end = datetime.datetime.now()
print((end-start).seconds)

