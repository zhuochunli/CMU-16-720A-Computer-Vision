import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
# from q2_2_sevenpoint import sevenpoint
from q3_1_essential_matrix import essentialMatrix
from q3_2_triangulate import findM2, triangulate

from scipy.optimize import leastsq

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # homography matrix
    pts1_hom = np.vstack((pts1.T, np.ones([1, pts1.shape[0]])))
    pts2_hom = np.vstack((pts2.T, np.ones([1, pts1.shape[0]])))

    max_inliers = 0
    inliers, best_F = None, None
    for i in range(nIters):
        # randomly choose 8 points
        rand_index = np.random.choice(pts1.shape[0], 8)
        rand_p1, rand_p2 = pts1[rand_index, :], pts2[rand_index, :]

        # get predicted points
        F = eightpoint(rand_p1, rand_p2, M)
        pred_p2_hom = F @ pts1_hom
        pred_p2 = pred_p2_hom/np.linalg.norm(pred_p2_hom[:2, :], axis=0)

        # calculate error
        err = abs(np.sum(pts2_hom*pred_p2, axis=0))
        cur_inliers = (err<tol).T
        if cur_inliers[cur_inliers].shape[0] > max_inliers:
            max_inliers = cur_inliers[cur_inliers].shape[0]
            best_F = F
            inliers = cur_inliers

    return best_F, inliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    else:
        # R = I*cos(θ) + (1 − cos θ)*u@u.T + ux*sin(θ)
        u = r/theta
        ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        R = np.eye(3)*np.cos(theta)+(1-np.cos(theta))*(u @ u.T)+ux*np.sin(theta)
        return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R-R.T)/2
    ro = np.array([[A[2][1], A[0][2], A[1][0]]]).T
    s = np.linalg.norm(ro)
    c = (R[0][0]+R[1][1]+R[2][2]-1)/2

    r = None
    if s == 0. and c == 1.:
        r = np.zeros((3, 1))
    elif s == 0. and c == -1.:
        # v = a nonzero column of R + I
        R_I = R+np.eye(3)
        for i in range(3):
            if np.count_nonzero(R_I[:, i]) > 0:
                v = R_I[:, i]
                break

        u = v/np.linalg.norm(v)
        tmp_r = u*np.pi
        if (np.linalg.norm(tmp_r) == np.pi) and ((tmp_r[0][0] == tmp_r[1][0] == 0. and tmp_r[2][0] < 0) or (tmp_r[0][0] == 0. and tmp_r[1][0] < 0.) or (tmp_r[0][0] < 0.)):
            r = -tmp_r
        else:
            r = tmp_r
    else:
        u = ro/s
        theta = np.arctan2(s, c)
        r = u*theta

    return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P = x[:-6].reshape(-1, 3)
    r2 = x[-6:-3].reshape(3, 1)
    t2 = x[-3:].reshape(3, 1)

    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    P_hom = np.vstack((P.T, np.ones((1, P.shape[0]))))

    x1_hom = np.dot(C1, P_hom)
    x2_hom = np.dot(C2, P_hom)
    p1_hat = np.zeros((2, P_hom.shape[1]))
    p2_hat = np.zeros((2, P_hom.shape[1]))

    p1_hat[:2, :] = (x1_hom[:2, :] / x1_hom[2, :])
    p2_hat[:2, :] = (x2_hom[:2, :] / x2_hom[2, :])
    p1_hat = p1_hat.T
    p2_hat = p2_hat.T
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])])

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    R2_init = M2_init[:, :3]
    t2_init = M2_init[:, 3]
    r2_init = invRodrigues(R2_init)
    x_init = np.concatenate([P_init.flatten(), r2_init.flatten(), t2_init.flatten()])
    obj_start = np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x_init)**2)

    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x))
    x_optimized, obj_end = leastsq(func, x_init)
    P2 = x_optimized[:-6].reshape(-1, 3)
    r2 = x_optimized[-6:-3].reshape(3, 1)
    t2 = x_optimized[-3:].reshape(3, 1)
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P2, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('../data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=100)

    # Simple Tests to verify your implementation, Q5.1:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)

    # Simple Tests to verify your implementation, Q5.2:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    # Q5.3
    F = eightpoint(noisy_pts1[inliers, :], noisy_pts2[inliers, :], M=np.max([*im1.shape, *im2.shape]))
    E = essentialMatrix(F, K1, K2)
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    C1 = K1 @ M1
    M2, C2, P = findM2(F, noisy_pts1, noisy_pts2, intrinsics, None)
    P_init, err = triangulate(C1, noisy_pts1[inliers, :], C2, noisy_pts2[inliers, :])
    M2, P2, obj_start, obj_end = bundleAdjustment(K1, M1, noisy_pts1[inliers, :], K2, M2, noisy_pts2[inliers, :], P_init)
    print('The initial reprojection error:', obj_start)
    print('The optimized reprojection error:', obj_end)
    plot_3D_dual(P_init, P2)