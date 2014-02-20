"""
Routines for fitting gaussian profiles to data
author: Niru Maheswaranathan
02:57 PM Feb 8, 2014
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from math import atan2
from matplotlib.patches import Ellipse




def test():
    x = [5,7,11,15,16,17,18]
    y = [8, 5, 8, 9, 17, 18, 25]
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt
    ax = plt.subplot(111, aspect='equal')
    for j in xrange(1, 4):
        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                      angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_facecolor('cyan')
        ell.set_alpha(0.1)
        ell.set_edgecolor('black')
        ax.add_artist(ell)
    plt.scatter(x, y)
    plt.show()
