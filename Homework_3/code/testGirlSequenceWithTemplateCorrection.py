import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=0.01, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
template_threshold = args.template_threshold
    
rects_red = np.load("../data/girlseqrects.npy")     # the origin tangle in Q1.3
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]
rects = np.zeros((seq.shape[2], 4))
rects[0, :] = rect

origin = rect
frame_temp = seq[:, :, 0]
frame_prev = frame_temp

p0 = np.zeros(2)
delta_p = np.zeros(2)

for i in range(1, seq.shape[2]):
    cur_frame = seq[:, :, i]

    p = LucasKanade(frame_prev, cur_frame, rect, threshold, num_iters, p0)  # pn
    p0 = np.copy(p)
    delta_p = [rect[0] + p[0] - origin[0], rect[1] + p[1] - origin[1]]
    p_star = LucasKanade(frame_temp, cur_frame, origin, threshold, num_iters, delta_p)  # pn*

    if np.linalg.norm(p_star - p0) <= template_threshold:       # update rect
        tmp = p_star - np.array([rect[0] - origin[0], rect[1] - origin[1]])
        rect = [rect[0] + tmp[0], rect[1] + tmp[1], rect[2] + tmp[0], rect[3] + tmp[1]]
        frame_prev = cur_frame
        p0 = np.zeros(2)
    else:   # no change to rect
        p0 = p

    rects[i, :] = [rect[0] + p[0], rect[1] + p[1], rect[2] + p[0], rect[3] + p[1]]

    # plot the figure
    if i in [1, 20, 40, 60, 80]:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        old_rect = rects_red[i, :]  # rectangle in girlseqrects.npy
        new_rect = rects[i, :]
        rect_red = patches.Rectangle((old_rect[0], old_rect[1]), old_rect[2] - old_rect[0], old_rect[3] - old_rect[1], edgecolor='red', facecolor='None')
        rect_blue = patches.Rectangle((new_rect[0], new_rect[1]), new_rect[2] - new_rect[0], new_rect[3] - new_rect[1], edgecolor='blue', facecolor='None')
        plt.imshow(seq[:, :, i], cmap='gray')       # Plot the original image
        ax.add_patch(rect_red)      # plot red rectangle
        ax.add_patch(rect_blue)     # plot red rectangle
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()

np.save('../data/girlseqrects-wcrt.npy', rects)