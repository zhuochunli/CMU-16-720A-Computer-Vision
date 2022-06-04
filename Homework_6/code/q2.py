# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 27, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    u, s, vh = np.linalg.svd(I, full_matrices=False)
    s[3:] = 0
    s_hat = np.diag(s[:3])
    B = np.sqrt(s_hat) @ vh[:3, :]
    L = u[:, :3] @ np.sqrt(s_hat)
    L = L.T
    return B, L


def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    GT_B = np.linalg.inv(G.T) @ B

    Nt = enforceIntegrability(GT_B, s)
    albedos, normals = estimateAlbedosNormals(Nt)
    surface = estimateShape(normals, s)
    plotSurface(surface)


if __name__ == "__main__":
    I, L, s = loadData()
    B_hat, L_hat = estimatePseudonormalsUncalibrated(I)

    # Part 2 (b)
    albedos, normals = estimateAlbedosNormals(B_hat)
    displayAlbedosNormals(albedos, normals, s)
    # Part 2 (d)
    albedos, normals = estimateAlbedosNormals(B_hat)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (e)
    Nt = enforceIntegrability(B_hat, s)
    albedos, normals = estimateAlbedosNormals(Nt)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (f)
    plotBasRelief(B_hat, 0, 0, -10)
    plotBasRelief(B_hat, 0, 0, 10)
    plotBasRelief(B_hat, 0, -10, 1)
    plotBasRelief(B_hat, 0, 10, 1)
    plotBasRelief(B_hat, -10, 0, 1)
    plotBasRelief(B_hat, 10, 0, 1)

