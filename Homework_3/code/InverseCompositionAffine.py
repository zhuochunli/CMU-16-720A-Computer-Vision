import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    p = M.flatten()
    x_min, y_min, x_max, y_max = 0, 0, It.shape[1] - 1, It.shape[0] - 1
    It_spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1_spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It1)

    x_temp = np.arange(x_min, x_max + 0.1)
    y_temp = np.arange(y_min, y_max + 0.1)
    x_meshIt, y_meshIt = np.meshgrid(x_temp, y_temp)

    It_interpX = It_spline.ev(y_meshIt, x_meshIt, dx=0, dy=1).flatten()
    It_interpY = It_spline.ev(y_meshIt, x_meshIt, dx=1, dy=0).flatten()

    A = np.zeros((It_interpX.shape[0], 6))
    A[:, 0] = np.multiply(It_interpX, x_meshIt.flatten())
    A[:, 1] = np.multiply(It_interpX, y_meshIt.flatten())
    A[:, 2] = It_interpX
    A[:, 3] = np.multiply(It_interpY, x_meshIt.flatten())
    A[:, 4] = np.multiply(It_interpY, y_meshIt.flatten())
    A[:, 5] = It_interpY

    for i in range(num_iters):
        x_cur = p[0] * x_meshIt + p[1] * y_meshIt + p[2]
        y_cur = p[3] * x_meshIt + p[4] * y_meshIt + p[5]
        points = (x_cur > 0) & (x_cur < It1.shape[1]) & (y_cur > 0) & (y_cur < It1.shape[0])
        x_cur, y_cur = x_cur[points], y_cur[points]
        It1_interp = It1_spline.ev(y_cur, x_cur)

        A_star = A[points.flatten()]
        b = It1_interp.flatten() - It[points].flatten()
        b = A_star.T @ b

        delta_p = np.linalg.inv(A_star.T @ A_star) @ b

        M = np.vstack((np.reshape(p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M = np.zeros((3, 3))
        delta_M[0, :] = [1+delta_p[0], delta_p[1], delta_p[2]]
        delta_M[1, :] = [delta_p[3], 1+delta_p[4], delta_p[5]]
        delta_M[2, :] = [0, 0, 1]
        M = M @ np.linalg.inv(delta_M)

        p = M[:2, :].flatten()
        if np.sum(delta_p ** 2) <= threshold:
            break

    M = M[:2, :]

    return M
