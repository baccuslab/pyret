"""
subspace tools
author: Niru Maheswaranathan
08:22 PM Feb 16, 2014
"""
import numpy as np
from numpy.linalg import qr, svd

def principalAngles(U, V):

    # compute the QR decomposition
    Qu, Ru = qr(U)
    Qv, Rv = qr(V)

    # find the singular values
    magnitude = svd(Qu.T.dot(Qv), compute_uv=False, full_matrices=False)

    # and the corresponding angles
    angles = np.rad2deg(np.arccos(magnitude))

    return angles, magnitude
