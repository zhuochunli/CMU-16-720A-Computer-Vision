import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    # ----- TODO -----
    # get [x2, y2]
    x1, y1 = temple_pts1['x1'], temple_pts1['y1']
    points1, points2 = [], []
    for i in range(x1.shape[0]):
        tmp_x2, tmp_y2 = epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
        points1.append([x1[i][0], y1[i][0]])
        points2.append([tmp_x2, tmp_y2])
    points1, points2 = np.array(points1), np.array(points2)

    M2, C2, P = findM2(F, points1, points2, intrinsics, '../data/q4_2.npz')

    return P


'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('../data/templeCoords.npz')
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    P = compute3D_pts(temple_coords_path, intrinsics, F, im1, im2)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
