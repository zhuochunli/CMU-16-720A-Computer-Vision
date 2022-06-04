import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Normalize the input pts1 and pts2 using the matrix T.
    pts1, pts2 = pts1/M, pts2/M
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    A = np.array((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(pts1.shape[0]))).T

    # Solve for the least square solution using SVD.
    u, s, vh = np.linalg.svd(A)
    F = vh[-1].reshape(3, 3)

    # Use the function `_singularize` (provided) to enforce the singularity condition.
    F = _singularize(F)

    # Use the function `refineF` (provided) to refine the computed fundamental matrix.
    F = refineF(F, pts1, pts2)

    # Unscale the fundamental matrix
    F = T.T @ F @ T

    return F


if __name__ == "__main__":
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # Q2.1
    # Write your code here
    np.savez('../data/q2_1.npz', F=F, M=np.max([*im1.shape, *im2.shape]))
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)