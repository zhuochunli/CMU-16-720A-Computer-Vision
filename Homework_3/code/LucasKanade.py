import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0
    x_min, y_min, x_max, y_max = rect[0], rect[1], rect[2], rect[3]
    It_spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1_spline = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    x_temp = np.arange(x_min, x_max + 0.1)
    y_temp = np.arange(y_min, y_max + 0.1)
    x_meshIt, y_meshIt = np.meshgrid(x_temp, y_temp)
    It_interp = It_spline.ev(y_meshIt, x_meshIt)

    for i in range(num_iters):
        x_cur = np.arange(x_min + p[0], x_max + p[0] + 0.1)
        y_cur = np.arange(y_min + p[1], y_max + p[1] + 0.1)
        x_meshIt1, y_meshIt1 = np.meshgrid(x_cur, y_cur)
        It1_interp = It1_spline.ev(y_meshIt1, x_meshIt1)

        # calculate gradients dx, dy
        It1_interpX = It1_spline.ev(y_meshIt1, x_meshIt1, dx=0, dy=1)
        It1_interpY = It1_spline.ev(y_meshIt1, x_meshIt1, dx=1, dy=0)
        A = np.vstack((It1_interpX.flatten(), It1_interpY.flatten())).T
        b = It_interp.flatten() - It1_interp.flatten()

        # update p
        H = A.T @ A
        delta_p = np.linalg.inv(H) @ (A.T @ b)
        p[0], p[1] = p[0]+delta_p[0], p[1]+delta_p[1]

        if np.sum(delta_p**2) < threshold:
            break

    return p
