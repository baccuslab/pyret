import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import common.spikeTriggeredAnalysis as st
from matplotlib import animation
from helpers.psth import computePSTH

"""
tools to plot spike rasters
author: Niru Maheswaranathan
4:58 PM Jan 7, 2013
"""

def plotRaster(firings):
    """
    Plots a spike raster, given a firings matrix
    """

    # create the figure
    fig = plt.figure()
    fig.clf()
    sns.set(style="whitegrid")
    ax = fig.add_subplot(111)

    # add the psth
    time,psth = computePSTH(firings)
    ax.plot(time, psth,
            linestyle='-',
            linewidth=2,
            color='gray')

    # add the rasters
    ax.plot(firings[:,0], firings[:,1], marker='o', linestyle='None', markersize=6, color='black')

    return fig, ax

def playSTA(sta):
    """
    plays a spatiotemporal STA movie
    """

    # initial frame
    im0 = sta[:,:,0]

    # set up the figure
    fig = plt.figure()
    ax = plt.axes(xlim=(0,sta.shape[0]), ylim=(0,sta.shape[1]))
    img = plt.imshow(im0)

    maxval = np.ceil(np.max(np.abs(sta)))
    img.set_cmap('gray')
    #img.set_clim(-maxval,maxval)
    img.set_interpolation('nearest')
    plt.colorbar()

    # initialization function
    def init():
        img.set_data(im0)
        return img

    # animation function (called sequentially)
    def animate(i):
        ax.set_title('Time %i' % i)
        img.set_data(sta[:,:,i])
        return img

    # call the animator
    anim = animation.FuncAnimation(fig, animate, np.arange(sta.shape[-1]), init_func=init, interval=50)
    plt.show()
    plt.draw()

def plotSTA(sta, timeSlice=-1):

    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # temporal slice to plot
    if timeSlice == -1:
        idx, spatialIdx, timeSlice = st.findFilterPeak(sta)

    # make the plot plot
    maxval = np.ceil(np.max(np.abs(sta)))
    imgplot = plt.imshow(sta[:,:,timeSlice])
    imgplot.set_cmap('RdBu')
    #imgplot.set_clim(-maxval,maxval)
    imgplot.set_interpolation('nearest')
    plt.colorbar()
    plt.show()
    plt.draw()
