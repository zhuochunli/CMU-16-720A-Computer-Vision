import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform


def LucasKanadeAffine(It, It1, threshold, num_iters):
    # Input:
    # 	It: template image
    # 	It1: Current image
    #  Output:
    # 	M: the Affine warp matrix [2x3 numpy array]
    #   put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    p = M.flatten()
    It1_spline = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    x_min, y_min, x_max, y_max = 0, 0, It.shape[1]-1, It.shape[0]-1
    x_temp = np.arange(x_min, x_max + 0.1)
    y_temp = np.arange(y_min, y_max + 0.1)
    x_meshIt1, y_meshIt1 = np.meshgrid(x_temp, y_temp)

    for i in range(num_iters):
        x_cur = p[0]*x_meshIt1 + p[1]*y_meshIt1 + p[2]
        y_cur = p[3]*x_meshIt1 + p[4]*y_meshIt1 + p[5]
        # find valid points
        points = (x_cur > 0) & (x_cur < It.shape[1]) & (y_cur > 0) & (y_cur < It.shape[0])
        x_cur, y_cur = x_cur[points], y_cur[points]
        It1_interp = It1_spline.ev(y_cur, x_cur)

        It1_interpX = It1_spline.ev(y_cur, x_cur, dx=0, dy=1).flatten()
        It1_interpY = It1_spline.ev(y_cur, x_cur, dx=1, dy=0).flatten()

        # calculate Affine matrix
        A = np.zeros((It1_interpX.shape[0], 6))
        A[:, 0] = np.multiply(It1_interpX, x_meshIt1[points].flatten())
        A[:, 1] = np.multiply(It1_interpX, y_meshIt1[points].flatten())
        A[:, 2] = It1_interpX
        A[:, 3] = np.multiply(It1_interpY, x_meshIt1[points].flatten())
        A[:, 4] = np.multiply(It1_interpY, y_meshIt1[points].flatten())
        A[:, 5] = It1_interpY

        # calculate matrix b
        b = It[points].flatten() - It1_interp.flatten()
        delta_p = np.linalg.inv(A.T @ A) @ (A.T @ b)
        p += delta_p.flatten()

        if np.sum(delta_p ** 2) <= threshold:
            break

    M = np.reshape(p, (2, 3))
    return M