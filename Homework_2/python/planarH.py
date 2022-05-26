import numpy as np
import cv2
from opts import get_opts

def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    A = np.zeros((x1.shape[0]*2, 9))
    for i in range(x1.shape[0]):
        u1, v1 = x1[i,0], x1[i,1]
        u2, v2 = x2[i,0], x2[i,1]
        # [xi, yi, 1, 0, 0, 0, -xi*ui, -yi*ui, -ui]
        A[i*2, :] = [u2, v2, 1, 0, 0, 0, -u2*u1, -v2*u1, -u1]
        # [0, 0, 0, xi, yi, 1, -xi*vi, -yi*vi, -vi]
        A[i*2+1, :] = [0, 0, 0, u2, v2, 1, -u2*v1, -v2*v1, -v1]

    # remove inf or NAN to prevent SVD couldn't converge
    A[np.isinf(A)] = 0
    A[np.isnan(A)] = 0
    u, s, vh = np.linalg.svd(A)
    # Vector(s) with the singular values, within each vector sorted in descending order.
    # So the eig_vector corresponding to the min eig_value lies in the last of vh
    min_eigVector = vh[-1]
    H2to1 = min_eigVector.reshape(3, 3)
    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    x1_centerX = np.average(x1[:, 0])
    x1_centerY = np.average(x1[:, 1])
    x2_centerX = np.average(x2[:, 0])
    x2_centerY = np.average(x2[:, 1])

    # Shift the origin of the points to the centroid
    x1_ori, x2_ori = np.zeros((x1.shape[0], 2)), np.zeros((x2.shape[0], 2))
    x1_ori[:, 0] = x1[:, 0] - x1_centerX
    x1_ori[:, 1] = x1[:, 1] - x1_centerY
    x2_ori[:, 0] = x2[:, 0] - x2_centerX
    x2_ori[:, 1] = x2[:, 1] - x2_centerY

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_maxDist = np.max(np.linalg.norm(x1_ori, axis=1))
    x2_maxDist = np.max(np.linalg.norm(x2_ori, axis=1))
    s1, s2 = np.sqrt(2)/x1_maxDist, np.sqrt(2)/x2_maxDist
    T1 = [[s1, 0, -s1*x1_centerX], [0, s1, -s1*x1_centerY], [0, 0, 1]]
    T2 = [[s2, 0, -s2*x2_centerX], [0, s2, -s2 * x2_centerY], [0, 0, 1]]

    # Similarity transform 1
    x1_h = np.concatenate((x1, np.ones((x1.shape[0], 1))), axis=1)
    x1_norm = np.zeros((x1.shape[0], 2))
    for i in range(x1.shape[0]):
        x1_norm[i] = np.matmul(T1, x1_h[i])[0:2]

    # Similarity transform 2
    x2_h = np.concatenate((x2, np.ones((x2.shape[0], 1))), axis=1)
    x2_norm = np.zeros((x2.shape[0], 2))
    for i in range(x2.shape[0]):
        x2_norm[i] = np.matmul(T2, x2_h[i])[0:2]

    # Compute homography
    H2to1_h = computeH(x1_norm, x2_norm)
    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H2to1_h @ T2
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol

    inliers = np.zeros(locs1.shape[0])
    max_inliers = float('-inf')     # the max number of inliers
    bestH2to1 = np.zeros((3, 3))
    predict_h = np.zeros((locs1.shape[0], 3))
    predict = np.zeros((locs1.shape[0], 2))     # predict points coordinates
    for i in range(max_iters):
        # at least 4 point pairs needed to compute H
        ran_rows = np.random.choice(locs1.shape[0], 4)
        tmp_h = computeH_norm(locs1[ran_rows, :], locs2[ran_rows, :])

        for j in range(locs1.shape[0]):
            predict_h[j, :] = np.matmul(tmp_h, np.append(locs2[j, :], 1))
            predict[j, :] = (predict_h[j, :]/predict_h[j, 2])[0:2]

        # remove possible inf or NAN caused by divide 0
        predict[np.isinf(predict)] = 0
        predict[np.isnan(predict)] = 0
        bias = np.linalg.norm(locs1 - predict, axis=1)
        index = np.where(bias <= inlier_tol)
        if index[0].shape[0] > max_inliers:
            max_inliers = index[0].shape[0]
            for point in index:
                inliers[point] = 1
            bestH2to1 = tmp_h

    return bestH2to1, inliers


def compositeH(H2to1, template, img):

    # Crea te a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    mask = np.ones(template.shape)

    # Warp mask by appropriate homography
    warp_mask = cv2.warpPerspective(mask, np.linalg.inv(H2to1), (img.shape[1], img.shape[0]))

    # Warp template by appropriate homography
    warp_template = cv2.warpPerspective(template, np.linalg.inv(H2to1), (img.shape[1], img.shape[0]))

    # Use mask to combine the warped template and the image
    index = np.all(warp_mask, axis=2)      # find the index we need replace (warp_mask=1)
    img[index, :] = warp_template[index, :]
    composite_img = img

    return composite_img
