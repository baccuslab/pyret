"""
Visualization functions for displaying spikes, filters, and cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from . import filtertools as ft
from .utils import plotwrapper
from matplotlib import gridspec, animation, cm
from matplotlib.patches import Ellipse

__all__ = ['raster', 'psth', 'rasterandpsth', 'spatial', 'temporal',
           'plotsta', 'playsta', 'ellipse', 'plotcells', 'playrates']


@plotwrapper
def raster(spikes, labels, title='Spike raster', marker_string='ko', **kwargs):
    """
    Plot a raster of spike times

    Parameters
    ----------
    spikes : array_like
        An array of spike times

    labels : array_like
        An array of labels corresponding to each spike in spikes. For example,
        this can indicate which cell or trial each spike came from

    title : string, optional
        An optional title for the plot (Default: 'Spike raster').

    marker_string : string, optional
        The marker string passed to matplotlib's plot function (Default: 'ko').

    kwargs : dict
        Optional keyword arguments are passed to matplotlib's plot function

    Returns
    -------
    fig : matplotlib Figure object
        Matplotlib handle of the figure

    ax : matplotlib Axes handle
    """
    fig, ax = kwargs['fig'], kwargs['ax']

    # data checking
    assert len(spikes) == len(labels), "Spikes and labels must have the same length"

    # Make a new figure
    if not fig or type(fig) is not plt.Figure:
        fig = plt.figure()

    # Plot the spikes
    if not ax:
        ax = fig.add_subplot(111)
    ax.plot(spikes, labels, marker_string, **kwargs)

    # Labels, etc.
    plt.title(title, fontdict={'fontsize': 24})
    plt.xlabel('Time (s)', fontdict={'fontsize': 20})


@plotwrapper
def psth(spikes, trial_length=None, binsize=0.01, **kwargs):
    """
    Plot a PSTH from the given spike times.

    Parameters
    ----------
    spikes : array_like
        An array of spike times

    triallength : float
        The length of each trial to stack, in seconds

    binsize : float
        The size of bins used in computing the PSTH

    fig : matplotlib Figure object
        Figure into which the psth should be plotted

    Returns
    -------
    fig : matplotlib Figure object
        Matplotlib figure handle

    """
    fig, ax = kwargs['fig'], kwargs['ax']

    # Input-checking
    if not trial_length:
        trial_length = spikes.max()

    # Compute the histogram bins to use
    ntrials = int(np.ceil(spikes.max() / trial_length))
    basebins = np.arange(0, trial_length + binsize, binsize)
    tbins = np.tile(basebins, (ntrials, 1)) + (np.tile(np.arange(0, ntrials), (basebins.size, 1)).T * trial_length)

    # Bin the spikes in each time bin
    bspk = np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = np.mean(bspk, axis=0) / binsize

    # Make a figure
    if not fig or type(fig) is not plt.Figure:
        fig = plt.figure()

    # Plot the PSTH
    ax = fig.add_subplot(111)
    ax.plot(tbins[0, :-1], fr, color='k', marker=None, linestyle='-', linewidth=2)

    # Labels etc
    plt.title('psth', fontsize=24)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('firing rate (Hz)', fontsize=20)


@plotwrapper
def rasterandpsth(spikes, trial_length=None, binsize=0.01, **kwargs):
    """
    Plot a spike raster and a PSTH on the same set of axes.

    Parameters
    ----------
    spikes : array_like
        An array of spike times

    triallength : float
        The length of each trial to stack, in seconds

    binsize : float
        The size of bins used in computing the PSTH

    fig : matplotlib Figure handle
        Figure into which the psth should be plotted

    Returns
    -------
    fig : matplotlib Figure handle
        Matplotlib figure handle
    """
    fig, ax = kwargs['fig'], kwargs['ax']

    # Input-checking
    if not trial_length:
        trial_length = spikes.max()

    # Compute the histogram bins to use
    ntrials = int(np.ceil(spikes.max() / trial_length))
    basebins = np.arange(0, trial_length + binsize, binsize)
    tbins = np.tile(basebins, (ntrials, 1)) + (np.tile(np.arange(0, ntrials), (basebins.size, 1)).T * trial_length)

    # Bin the spikes in each time bin
    bspk = np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = np.mean(bspk, axis=0) / binsize

    # Plot the PSTH
    psthax = fig.add_subplot(111)
    psthax.plot(tbins[0, :-1], fr, color='r', marker=None, linestyle='-', linewidth=2)
    psthax.set_title('psth and raster', fontdict={'fontsize':24})
    psthax.set_xlabel('time (s)', fontdict={'fontsize':20})
    psthax.set_ylabel('firing rate (Hz)', color='r', fontdict={'fontsize':20})
    for tick in psthax.get_yticklabels():
        tick.set_color('r')

    # Plot the raster
    rastax = psthax.twinx()
    plt.hold(True)
    for trial in range(ntrials):
        idx = np.bitwise_and(spikes > tbins[trial, 0], spikes <= tbins[trial, -1])
        rastax.plot(spikes[idx] - tbins[trial, 0], trial * np.ones(spikes[idx].shape),
                    color='k', marker='.', linestyle='none')
    rastax.set_ylabel('trial #', color='k', fontdict={'fontsize':20})
    for tick in psthax.get_yticklabels():
        tick.set_color('k')


def playsta(sta, repeat=True, frametime=100, cmap='seismic_r', clim=None):
    """
    Plays a spatiotemporal spike-triggered average as a movie

    Parameters
    ----------
    sta : array_like
        Spike-triggered average array, shaped as (nframes, npix, npix)

    repeat : boolean, optional
        Whether or not to repeat the animation (default is True)

    frametime : float, optional
        Length of time each frame is displayed for in milliseconds (default is 100)

    cmap : string, optional
        Name of the colormap to use (Default: gray)

    clim : array_like, optional
        2 Dimensional color limit for animation; e.g. [0, 255]

    Returns
    -------
    anim : matplotlib animation object
    """
    # mean subtract
    X = sta.copy()
    X -= X.mean()

    # Initial frame
    initial_frame = X[0]

    # Set up the figure
    fig = plt.figure()
    plt.axis('equal')
    ax = plt.axes(xlim=(0, X.shape[1]), ylim=(0, X.shape[2]))
    img = plt.imshow(initial_frame)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set up the colors
    img.set_cmap(cmap)
    img.set_interpolation('nearest')
    if clim is not None:
        img.set_clim(clim)
    else:
        maxval = np.max(np.abs(X))
        img.set_clim([-maxval, maxval])

    # Animation function (called sequentially)
    def animate(i):
        ax.set_title('Frame {0:#d}'.format(i + 1))
        img.set_data(X[i])

    # Call the animator
    anim = animation.FuncAnimation(fig, animate, np.arange(X.shape[0]),
                                   interval=frametime, repeat=repeat)
    plt.show()
    plt.draw()

    return anim


@plotwrapper
def spatial(spatial_filter, maxval=None, **kwargs):
    """
    Plot a spatial filter on a given axes

    Parameters
    ----------
    spatialFrame : array_like
        The frame to plot, as an (n x n) matrix.

    ax : matplotlib Axes object, optional
        the axes on which to plot the data; defaults to creating a new figure

    clim : tuple, optional
        the color range with which to scale the image. Defaults to [-maxval, maxval] where maxval is the max abs. value

    Returns
    -------
    ax : matplotlib Axes object
        Axes into which the frame is plotted
    """
    _, ax = kwargs.pop('fig'), kwargs.pop('ax')

    # adjust color limits if necessary
    if not maxval:

        # normalize
        spatial_filter -= np.mean(spatial_filter)

        # find max abs value
        maxval = np.max(np.abs(spatial_filter))

    # plot the spatial frame
    ax.imshow(spatial_filter,
              cmap='seismic_r',
              interpolation='nearest',
              aspect='equal',
              vmin=-maxval,
              vmax=maxval,
              **kwargs)


@plotwrapper
def temporal(time, temporal_filter, **kwargs):
    """
    Plot a temporal filter on a given axes

    Parameters
    ----------
    time : array_like
        a time vector to plot against

    temporal_filter : ndarray
        the temporal filter to plot, has the same dimensions as time

    ax : matplotlib Axes object, optional
        the axes on which to plot the data; defaults to creating a new figure

    Returns
    -------
    ax : matplotlib Axes object
        Axes into which the frame is plotted
    """
    kwargs['ax'].plot(time, temporal_filter, linestyle='-', linewidth=2, color='LightCoral')


def plotsta(time, sta):
    """
    Plot a spatial and temporal filter

    Parameters
    ----------
    time : array_like
        a time vector to plot against

    sta : array_like
        the filter to plot

    timeslice : int, optional
        the index of the spatial slice to plot

    Returns
    -------
    ax : matplotlib Axes object
        Axes into which the STA is plotted
    """

    # plot 1D temporal filter
    if sta.ndim == 1:
        fig = plt.figure(figsize=(12, 8))
        fig, ax = temporal(time, sta, ax=fig.add_subplot(111))

    # plot 2D spatiotemporal filter
    elif sta.ndim == 2:

        # normalize
        stan = (sta - np.mean(sta)) / np.var(sta)

        # create new axes
        fig = plt.figure(figsize=(10, 10))
        fig, ax = spatial(stan, ax=fig.add_subplot(111))
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

    # plot 3D spatiotemporal filter
    elif sta.ndim == 3:

        # build the figure
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # decompose
        spatial_profile, temporal_filter = ft.decompose(sta)

        # plot spatial profile
        _, axspatial = spatial(spatial_profile, ax=fig.add_subplot(gs[0]))
        axspatial.set_xticks([])
        axspatial.set_yticks([])

        # plot temporal profile
        fig, axtemporal = temporal(time, temporal_filter, ax=fig.add_subplot(gs[1]))
        axtemporal.set_xlim(time[0], time[-1])
        axtemporal.spines['right'].set_color('none')
        axtemporal.spines['top'].set_color('none')
        axtemporal.yaxis.set_ticks_position('left')
        axtemporal.xaxis.set_ticks_position('bottom')

        # return handles
        ax = (axspatial, axtemporal)

    else:
        raise ValueError('The sta parameter has an invalid number of dimensions (must be 1-3)')

    return fig, ax


@plotwrapper
def ellipse(spatial_filter, pvalue=0.6827, alpha=0.8, fc='none', ec='black', lw=3, **kwargs):
    """
    Plot a given ellipse

    Parameters
    ----------
    spatial_filter : array_like
        A spatial filter (2D image) corresponding to the spatial profile of the
        receptive field

    pvalue : float, optional
        Determines the threshold of the ellipse contours. For example, a pvalue
        of 0.95 corresponds to a 95% confidence ellipse. (Default: 0.6827)

    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque) (Default: 0.8)

    fc : string, optional
        Ellipse face color. (Default: none)

    ec : string, optional
        Ellipse edge color. (Default: black)

    lw : int, optional
        Line width. (Default: 3)

    ax : matplotlib Axes object, optional
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    Returns
    -------
    ax : matplotlib Axes object
        The axes onto which the ellipse is plotted
    """
    ax = kwargs['ax']

    # get the ellipse parameters
    center, widths, theta = ft.get_ellipse(np.arange(spatial_filter.shape[0]),
                                           np.arange(spatial_filter.shape[1]),
                                           spatial_filter,
                                           pvalue=pvalue)

    # create the ellipse
    ell = Ellipse(xy=center, width=widths[0], height=widths[1], angle=theta,
                  alpha=alpha, ec=ec, fc=fc, lw=lw, **kwargs)

    ax.add_artist(ell)
    ax.set_xlim(0, spatial_filter.shape[0])
    ax.set_ylim(0, spatial_filter.shape[1])


@plotwrapper
def plotcells(cells, **kwargs):
    """
    Plot the spatial receptive fields for multiple cells

    Parameters
    ----------
    cells : list of array_like
        A list of spatiotemporal receptive fields, each of which is a spatiotemporal array

    ax : matplotlib Axes object, optional
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    Returns
    ------
    ax : matplotlib Axes object
        The axes onto which the ellipse is plotted
    """
    ax = kwargs['ax']

    # for each cell
    for idx, sta in enumerate(cells):

        # get the spatial profile
        sp = ft.decompose(sta)[0]

        # plot ellipse
        color = cm.Set1(np.random.randint(100))
        ax = ellipse(sp, fc=color, ec=color, lw=2, alpha=0.3, ax=ax)


def playrates(rates, patches, num_levels=255, time=None, repeat=True, frametime=100):
    """
    Plays a movie of the firing rate for N cells by updating the given patches (matplotlib handles)
    (useful in conjunction with the output of plotcells)

    Parameters
    ----------
    rates : array_like
        an (N, T) matrix of firing rates

    patches : list
        A list of N matplotlib patch elements. The facecolor of these patches is altered according to the rates values.

    Returns
    -------
    anim : matplotlib Animation object
    """
    # approximate necessary colormap
    colors = cm.gray(np.arange(num_levels))
    rscale = np.round((num_levels - 1) * (rates - rates.min()) /
                      (rates.max() - rates.min())).astype('int')

    # set up
    fig = plt.gcf()
    ax = plt.gca()
    if time is None:
        time = np.arange(rscale.shape[1])

    # Animation function (called sequentially)
    def animate(t):
        for i in range(rscale.shape[0]):
            patches[i].set_facecolor(colors[rscale[i, t]])
        ax.set_title('Time: %0.2f seconds' % (time[t]), fontsize=20)

    # Call the animator
    anim = animation.FuncAnimation(fig, animate, np.arange(rscale.shape[1]), interval=frametime, repeat=repeat)
    return anim
