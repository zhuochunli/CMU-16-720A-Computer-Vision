import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
rects = np.zeros((seq.shape[2], 4))
rects[0, :] = rect
total_p = np.zeros(2)
for i in range(1, seq.shape[2]):
    p = LucasKanade(seq[:, :, i - 1], seq[:, :, i], np.transpose(rects[i - 1, :]), threshold, num_iters)
    total_p += p
    rects[i, :] = [59 + total_p[0], 116 + total_p[1], 145 + total_p[0], 151 + total_p[1]]
    if i in [1, 100, 200, 300, 400]:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cur_rect = rects[i, :]
        rectangle = patches.Rectangle((cur_rect[0], cur_rect[1]), cur_rect[2] - cur_rect[0], cur_rect[3] - cur_rect[1], edgecolor='red', facecolor='None')
        plt.imshow(seq[:, :, i], cmap='gray')
        ax.add_patch(rectangle)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()

np.save('../data/carseqrects.npy', rects)
