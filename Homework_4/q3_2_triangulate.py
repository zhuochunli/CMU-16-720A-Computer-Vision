import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):
    # For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    A1 = np.array((C1[0, 0]-C1[2, 0]*x1, C1[0, 1]-C1[2, 1]*x1, C1[0, 2]-C1[2, 2]*x1, C1[0, 3]-C1[2, 3]*x1)).T
    A2 = np.array((C1[1, 0] - C1[2, 0] * y1, C1[1, 1] - C1[2, 1] * y1, C1[1, 2] - C1[2, 2] * y1, C1[1, 3] - C1[2, 3] * y1)).T
    A3 = np.array((C2[0, 0] - C2[2, 0] * x2, C2[0, 1] - C2[2, 1] * x2, C2[0, 2] - C2[2, 2] * x2, C2[0, 3] - C2[2, 3] * x2)).T
    A4 = np.array((C2[1, 0] - C2[2, 0] * y2, C2[1, 1] - C2[2, 1] * y2, C2[1, 2] - C2[2, 2] * y2, C2[1, 3] - C2[2, 3] * y2)).T

    w = np.zeros((pts1.shape[0], 3))    # wi = [xi, yi, zi].T, Nx3
    for i in range(pts1.shape[0]):
        A = np.vstack((A1[i, :], A2[i, :], A3[i, :], A4[i, :]))     # 4x4
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        w[i, :] = [p[0]/p[-1], p[1]/p[-1], p[2]/p[-1]]

    # convert w from homogeneous coordinates to non-homogeneous ones, 4x1
    w_homo = np.hstack((w, np.ones((pts1.shape[0], 1))))
    err = 0
    for i in range(pts1.shape[0]):
        proj1 = C1 @ w_homo[i, :].T
        proj2 = C2 @ w_homo[i, :].T
        proj1 = (proj1[:2]/proj1[-1]).T
        proj2 = (proj2[:2]/proj2[-1]).T
        # Calculate the reprojection error
        err += np.sum((pts1[i]-proj1)**2+(pts2[i]-proj2)**2)

    return w, err

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename='../data/q3_3.npz'):
    '''
    Q3.3: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    C1 = K1 @ M1

    err = float('inf')
    M2, C2, P = None, None, None
    for i in range(M2s.shape[2]):
        cur_C2 = K2 @ M2s[:, :, i]
        cur_w, cur_err = triangulate(C1, pts1, cur_C2, pts2)

        if cur_err < err and np.min(cur_w[:, 2]) >= 0:     # valid points with all coordinates>=0
            err = cur_err
            M2 = M2s[:, :, i]
            C2 = cur_C2
            P = cur_w

    if filename == '../data/q4_2.npz':
        np.savez(filename, F=F, M1=M1, M2=M2, C1=C1, C2=C2)
    elif filename == '../data/q3_3.npz':
        np.savez(filename, M2=M2, C2=C2, P=P)
    return M2, C2, P


if __name__ == "__main__":

    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert(err < 500)