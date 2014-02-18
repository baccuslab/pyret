"""
Event handlers for interacting with figures
author: Niru Maheswaranathan
02:43 AM Jan 30, 2014
"""

import numpy as np
import matplotlib.pyplot as plt

def printPoints(xpts, ypts):
    """
    Print points as you click on them
    """

    # the pick event
    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        if len(ind) == 1:
            print('Point: (%f,%f)' % (xdata[ind[0]], ydata[ind[0]]))
        else:
            print('Multiple points clicked, try zooming in')

    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.canvas.mpl_connect('pick_event', onpick)

    # plot the points
    plt.plot(xpts, ypts, '.', picker=3)
    plt.show()
    plt.draw()
