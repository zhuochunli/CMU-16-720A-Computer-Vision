import numpy as np
import cv2
from helper import loadVid
from opts import get_opts
import matchPics
import planarH
import multiprocessing
import os
import datetime

opts = get_opts()
source = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')
output = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (book[0].shape[1], book[0].shape[0]))
output_length = min(len(source), len(book))


# Q3.1
def ar_process():
    output_frame = np.zeros((source.shape[1], source.shape[1]*cv_cover.shape[1]//cv_cover.shape[0]))
    offset = (source.shape[2] - output_frame.shape[1])//2
    list_args = []
    for i in range(output_length):     # use the shorter length for your video
        source_frame = source[i]
        book_frame = book[i]
        # crop the source frame
        output_frame = source_frame[:, offset:offset+output_frame.shape[1], :]
        list_args.append((book_frame, output_frame, i))

    p = multiprocessing.Pool(processes=8)
    p.map(single_process, list_args)

    for i in range(output_length):
        composite = np.load('../result/'+str(i)+'.npy')
        os.remove('../result/'+str(i)+'.npy')
        output.write(composite)
    output.release()


def single_process(args):   # similar as warpImage()
    book_frame, output_frame, index = args
    matches, locs1, locs2 = matchPics.matchPics(cv_cover, book_frame, opts)

    # scaling
    x1, x2 = np.zeros(locs1.shape), np.zeros(locs2.shape)
    x1[:, 0] = locs1[:, 1] * output_frame.shape[1] / cv_cover.shape[1]
    x1[:, 1] = locs1[:, 0] * output_frame.shape[0] / cv_cover.shape[0]
    x2[:, 0], x2[:, 1] = locs2[:, 1], locs2[:, 0]
    locs1, locs2 = x1[matches[:, 0]], x2[matches[:, 1]]

    # Use matching locs1 and locs2 to compute H
    bestH2to1, inliers = planarH.computeH_ransac(locs1, locs2, opts)

    # Use H to transform hp cover, with same output size as cv desk
    composite = planarH.compositeH(bestH2to1, output_frame, book_frame)
    np.save('../result/'+str(index), composite)


if __name__ == "__main__":
    start = datetime.datetime.now()
    ar_process()
    end = datetime.datetime.now()
    print('Time:', (end-start).total_seconds()//60, 'mins')