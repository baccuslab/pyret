"""
Visualization functions for displaying spikes, filters, and cells.

"""

import numpy as np
import matplotlib.pyplot as plt
from . import filtertools as ft
from matplotlib import animation as animation
from matplotlib.cm import Set1, gray

__all__ = ['raster', 'psth', 'rasterandpsth', 'spatial', 'temporal',
           'plotsta', 'playsta', 'ellipse', 'plotcells', 'playrates']


def raster(spikes, labels, title='Spike raster', marker_string='ko', fig=None, **kwargs):
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

    """

    # data checking
    assert len(spikes) == len(labels), "Spikes and labels must have the same length"

    # Make a new figure
    if not fig or type(fig) is not plt.Figure:
        fig = plt.figure()

    # Plot the spikes
    ax = fig.add_subplot(111)
    ax.plot(spikes, labels, marker_string, **kwargs)

    # Labels, etc.
    plt.title(title, fontdict={'fontsize': 24})
    plt.xlabel('Time (s)', fontdict={'fontsize': 20})
    plt.show()
    plt.draw()

    return fig


def psth(spikes, trial_length=None, binsize=0.01, fig=None):
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
    plt.title('psth', fontdict={'fontsize':24})
    plt.xlabel('time (s)', fontdict={'fontsize':20})
    plt.ylabel('firing rate (Hz)', fontdict={'fontsize':20})
    plt.show()
    plt.draw()

    return fig


def rasterandpsth(spikes, trial_length=None, binsize=0.01, fig=None):
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

    # Show the figure
    plt.show()
    plt.draw()

    return fig


def playsta(sta, repeat=True, frametime=100, cmap='seismic', clim=None):
    """
    Plays a spatiotemporal spike-triggered average as a movie

    Parameters
    ----------
    sta : array_like
        Spike-triggered average array, shaped as (npix, npix, nframes)

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
    None

    """

    # Initial frame
    initial_frame = sta[0]

    # Set up the figure
    fig = plt.figure()
    plt.axis('equal')
    ax = plt.axes(xlim=(0, sta.shape[1]), ylim=(0, sta.shape[2]))
    img = plt.imshow(initial_frame)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set up the colors
    img.set_cmap(cmap)
    img.set_interpolation('nearest')
    if clim is not None:
        img.set_clim(clim)
    else:
        maxval = np.max(np.abs(sta))
        img.set_clim([-maxval, maxval])

    # Animation function (called sequentially)
    def animate(i):
        ax.set_title('Frame {0:#d}'.format(i + 1))
        img.set_data(sta[i])

    # Call the animator
    anim = animation.FuncAnimation(fig, animate, np.arange(sta.shape[0]),
                                   interval=frametime, repeat=repeat)
    plt.show()
    plt.draw()

    return anim


def spatial(spatial_filter, ax=None, clim=None):
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

    if not ax:
        ax = plt.figure().add_subplot(111)

    # adjust color limits if necessary
    if not clim:

        # normalize
        spatial_filter -= np.mean(spatial_filter)

        # find max abs value
        maxabs = np.max(np.abs(spatial_filter))

        # set clim
        clim = (-maxabs, maxabs)

    # plot the spatial frame
    img = ax.imshow(spatial_filter, cmap='bwr', interpolation='nearest')
    img.set_clim(clim)
    ax.set_title('Spatial RF')
    ax.set_aspect('equal')

    # add colorbar
    # ax.get_figure().colorbar(img)

    plt.show()
    plt.draw()

    return ax


def temporal(time, temporal_filter, ax=None):
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

    if not ax:
        ax = plt.figure().add_subplot(111)

    ax.plot(time, temporal_filter, linestyle='-', linewidth=2, color='LightCoral')
    plt.show()
    plt.draw()

    return ax


def plotsta(time, sta, fig=None):
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

    # create the figure object
    if fig is None:
        fig = plt.figure()

    # plot 1D temporal filter
    if sta.ndim == 1:

        # plot temporal profile
        ax = temporal(time, sta, fig.add_subplot(111))

    # plot 2D spatiotemporal filter
    elif sta.ndim == 2:

        # normalize
        stan = (sta - np.mean(sta)) / np.var(sta)
        lim = np.max(np.abs(stan)) * 1.2

        # create new axes
        ax = fig.add_subplot(111)

        im = ax.imshow(stan)
        ax.set_aspect('equal')
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        im.set_clim(-lim, lim)
        im.set_cmap('seismic')
        plt.show()
        plt.draw()

    # plot 3D spatiotemporal filter
    elif sta.ndim == 3:

        # decompose
        spatial_profile, temporal_filter = ft.decompose(sta)

        # plot spatial profile
        axspatial = spatial(spatial_profile, fig.add_subplot(121))
        axspatial.set_xticks([])
        axspatial.set_yticks([])

        # plot temporal profile
        axtemporal = temporal(time, temporal_filter, fig.add_subplot(122))
        axtemporal.set_xlim(time[0], time[-1])

        # return handles
        ax = (axspatial, axtemporal)

    else:
        raise ValueError('The sta parameter has an invalid number of dimensions (must be 1-3)')

    return fig, ax


def ellipse(ell, ax=None):
    """
    Plot a given ellipse

    Parameters
    ----------
    ell : matplotlib.patches.Ellipse object
        The ellipse to be plotted

    ax : matplotlib Axes object, optional
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    Returns
    -------
    ax : matplotlib Axes object
        The axes onto which the ellipse is plotted

    """

    # Set some properties
    ell.set_facecolor('green')
    ell.set_alpha(0.5)
    ell.set_edgecolor('black')

    # Create axes or add to given
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.add_artist(ell)

    plt.show()
    plt.draw()
    return ax


def plotcells(cells, box, ax=None, scale=0.25):
    """
    Plot the spatial receptive fields for multiple cells

    Parameters
    ----------
    cells : list of array_like
        A list of spatiotemporal receptive fields, each of which is a spatiotemporal array

    ax : matplotlib Axes object, optional
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    box_dims : (float, float), optional
        Dimensions of a box (to indicate the electrode array) to draw behind the cells. Should be a tuple containing the
        (width,height) of the box. Defaults to

    start : float, optional
        Location of the lower left corner of the box to draw. If None, the box is centered on the plot. Only matters if
        `box_dims` is not None.

    Returns
    ------
    ax : matplotlib Axes object
        The axes onto which the ellipse is plotted

    """

    # Create axes or add to given
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # define the color palatte
    colors = Set1(np.arange(len(cells)))
    np.random.shuffle(colors)

    # for each cell
    ellipses = list()
    for idx, sta in enumerate(cells):

        # get the spatial profile
        _, _, tidx = ft.filterpeak(sta)

        # generate ellipse
        # tx = np.linspace(-1388.8888, 1388.8888, sta.shape[1])
        # ty = np.linspace(-1388.8888, 1388.8888, sta.shape[2])
        tx = np.arange(sta.shape[1])
        ty = np.arange(sta.shape[2])
        ell = ft.fit_ellipse(tx, ty, sta[tidx, :, :], scale=scale)

        # add it to the plot
        ell.set_facecolor(colors[idx])
        ell.set_edgecolor(colors[idx])
        ell.set_linewidth(2)
        ell.set_linestyle('solid')
        ell.set_alpha(0.3)
        ax.add_artist(ell)
        ellipses.append(ell)

    ax.add_patch(plt.Rectangle((10, 10), 30, 30,
                               fill=False, edgecolor='Black', linestyle='solid'))

    plt.xlim(xmin=box[0], xmax=box[1])
    plt.ylim(ymin=box[0], ymax=box[1])

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.box('off')
    plt.tight_layout()
    plt.show()
    plt.draw()
    return ax, ellipses


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
    colors = gray(np.arange(num_levels))
    rscale = np.round( (num_levels - 1) * (rates - rates.min()) / (rates.max() - rates.min()) ).astype('int')

    # set up
    fig = plt.gcf()
    ax = plt.gca()
    if time is None:
        time = np.arange(rscale.shape[1])

    # Animation function (called sequentially)
    def animate(t):
        for i in range(rscale.shape[0]):
            patches[i].set_facecolor(colors[rscale[i,t]])
        ax.set_title('Time: %0.2f seconds' % (time[t]), fontsize=20)

    # Call the animator
    anim = animation.FuncAnimation(fig, animate, np.arange(rscale.shape[1]), interval=frametime, repeat=repeat)
    plt.show()
    plt.draw()
    return anim
