"""
Visualization functions for displaying spikes, filters, and cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from . import filtertools as ft
from .utils import plotwrapper
from matplotlib import gridspec, animation, cm
from matplotlib.patches import Ellipse

__all__ = ['raster', 'psth', 'raster_and_psth', 'spatial', 'temporal',
           'plot_sta', 'play_sta', 'ellipse', 'plot_cells', 'play_rates']


@plotwrapper
def raster(spikes, labels, title='Spike raster', marker_string='ko', **kwargs):
    """
    Plot a raster of spike times.

    Parameters
    ----------
    spikes : array_like
        An array of spike times.

    labels : array_like
        An array of labels corresponding to each spike in spikes. For example,
        this can indicate which cell or trial each spike came from. Spike times
        are plotted on the x-axis, and labels on the y-axis.

    title : string, optional
        An optional title for the plot (Default: 'Spike raster').

    marker_string : string, optional
        The marker string passed to matplotlib's plot function (Default: 'ko').

    ax : matplotlib.axes.Axes instance, optional
        An optional axes onto which the data is plotted.

    fig : matplotlib.figure.Figure instance, optional
        An optional figure onto which the data is plotted.

    kwargs : dict
        Optional keyword arguments are passed to matplotlib's plot function

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object into which raster is plotted.

    ax : matplotlib.axes.Axes
        Matplotlib Axes object into which raster is plotted.
    """
    assert len(spikes) == len(labels), "Spikes and labels must have the same length"

    kwargs.pop('fig')
    ax = kwargs.pop('ax')

    # Plot the spikes
    ax.plot(spikes, labels, marker_string, **kwargs)

    # Labels, etc.
    ax.set_title(title, fontdict={'fontsize': 24})
    ax.set_xlabel('Time (s)', fontdict={'fontsize': 20})


@plotwrapper
def psth(spikes, trial_length=None, binsize=0.01, **kwargs):
    """
    Plot a PSTH from the given spike times.

    Parameters
    ----------
    spikes : array_like
        An array of spike times.

    trial_length : float
        The length of each trial to stack, in seconds. If None (the
        default), a single PSTH is plotted. If a float is passed, PSTHs
        from each trial of the given length are averaged together before
        plotting.

    binsize : float
        The size of bins used in computing the PSTH.

    ax : matplotlib.axes.Axes instance, optional
        An optional axes onto which the data is plotted.

    fig : matplotlib.figure.Figure instance, optional
        An optional figure onto which the data is plotted.

    kwargs : dict
        Keyword arguments passed to matplotlib's ``plot`` function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object into which PSTH is plotted.

    ax : matplotlib.axes.Axes
        Matplotlib Axes object into which PSTH is plotted.
    """
    fig = kwargs.pop('fig')
    ax = kwargs.pop('ax')

    # Input-checking
    if not trial_length:
        trial_length = spikes.max()

    # Compute the histogram bins to use
    ntrials = int(np.ceil(spikes.max() / trial_length))
    basebins = np.arange(0, trial_length + binsize, binsize)
    tbins = np.tile(basebins, (ntrials, 1)) + (np.tile(np.arange(0, ntrials), 
            (basebins.size, 1)).T * trial_length)

    # Bin the spikes in each time bin
    bspk = np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = np.mean(bspk, axis=0) / binsize

    # Plot the PSTH
    ax.plot(tbins[0, :-1], fr, color='k', marker=None, linestyle='-', linewidth=2)

    # Labels etc
    ax.set_title('PSTH', fontsize=24)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Firing rate (Hz)', fontsize=20)


@plotwrapper
def raster_and_psth(spikes, trial_length=None, binsize=0.01, **kwargs):
    """
    Plot a spike raster and a PSTH on the same set of axes.

    Parameters
    ----------
    spikes : array_like
        An array of spike times.

    trial_length : float
        The length of each trial to stack, in seconds. If None (the default),
        all spikes are plotted as part of the same trial.

    binsize : float
        The size of bins used in computing the PSTH.

    ax : matplotlib.axes.Axes instance, optional
        An optional axes onto which the data is plotted.

    fig : matplotlib.figure.Figure instance, optional
        An optional figure onto which the data is plotted.

    kwargs : dict
        Keyword arguments to matplotlib's ``plot`` function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure instance onto which the data is plotted.

    ax : matplotlib.axes.Axes
        Matplotlib Axes instance onto which the data is plotted.
    """
    fig = kwargs.pop('fig')
    ax = kwargs.pop('ax')

    # Input-checking
    if not trial_length:
        trial_length = spikes.max()

    # Compute the histogram bins to use
    ntrials = int(np.ceil(spikes.max() / trial_length))
    basebins = np.arange(0, trial_length + binsize, binsize)
    tbins = np.tile(basebins, (ntrials, 1)) + (np.tile(np.arange(0, ntrials), 
            (basebins.size, 1)).T * trial_length)

    # Bin the spikes in each time bin
    bspk = np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = np.mean(bspk, axis=0) / binsize

    # Plot the PSTH
    ax.plot(tbins[0, :-1], fr, color='r', marker=None, linestyle='-', linewidth=2)
    ax.set_xlabel('Time (s)', fontdict={'fontsize': 20})
    ax.set_ylabel('Firing rate (Hz)', color='r', fontdict={'fontsize': 20})
    for tick in ax.get_yticklabels():
        tick.set_color('r')

    # Plot the raster
    rastax = ax.twinx()
    plt.hold(True)
    for trial in range(ntrials):
        idx = np.bitwise_and(spikes > tbins[trial, 0], spikes <= tbins[trial, -1])
        rastax.plot(spikes[idx] - tbins[trial, 0], trial * np.ones(spikes[idx].shape),
                    color='k', marker='.', linestyle='none')
    rastax.set_ylabel('Trial #', color='k', fontdict={'fontsize': 20})
    for tick in ax.get_yticklabels():
        tick.set_color('k')


def play_sta(sta, repeat=True, frametime=100, cmap='seismic_r', clim=None):
    """
    Plays a spatiotemporal spike-triggered average as a movie.

    Parameters
    ----------
    sta : array_like
        Spike-triggered average array, shaped as ``(nt, nx, ny)``.

    repeat : boolean, optional
        Whether or not to repeat the animation (default is True).

    frametime : float, optional
        Length of time each frame is displayed for in milliseconds (default is 100).

    cmap : string, optional
        Name of the colormap to use (Default: ``'seismic_r'``).

    clim : array_like, optional
        2-element color limit for animation; e.g. [0, 255].

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
def spatial(filt, maxval=None, **kwargs):
    """
    Plot the spatial component of a full linear filter.

    If the given filter is 2D, it is assumed to be a 1D spatial filter,
    and is plotted directly. If the filter is 3D, it is decomposed into
    its spatial and temporal components, and the spatial component is plotted.

    Parameters
    ----------
    filt : array_like
        The filter whose spatial component is to be plotted. It may have
        temporal components.

    maxval : float, optional
        The value to use as minimal and maximal values when normalizing the
        colormap for this plot. See ``plt.imshow()`` documentation for more
        details.

    ax : matplotlib Axes object, optional
        The axes on which to plot the data; defaults to creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure onto which the spatial STA is plotted.

    ax : matplotlib Axes object
        Axes into which the spatial STA is plotted.
    """
    fig = kwargs.pop('fig')
    ax = kwargs.pop('ax')

    if filt.ndim > 2:
        spatial_filter, _ = ft.decompose(filt)
    else:
        spatial_filter = filt.copy()

    # adjust color limits if necessary
    if not maxval:
        spatial_filter -= np.mean(spatial_filter)
        maxval = np.max(np.abs(spatial_filter))

    # plot the spatial component
    ax.imshow(spatial_filter,
              cmap='seismic_r',
              interpolation='nearest',
              aspect='equal',
              vmin=-maxval,
              vmax=maxval,
              **kwargs)


@plotwrapper
def temporal(time, filt, **kwargs):
    """
    Plot the temporal component of a full linear filter.

    If the given linear filter is 1D, it is assumed to be a temporal filter,
    and is plotted directly. If the filter is 2 or 3D, it is decomposed into
    its spatial and temporal components, and the temporal component is plotted.

    Parameters
    ----------
    time : array_like
        A time vector to plot against.

    filt : array_like
        The full filter to plot. May be than 1D, but must match in size along
        the first dimension with the ``time`` input.

    ax : matplotlib Axes object, optional
        the axes on which to plot the data; defaults to creating a new figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure onto which the temoral STA is plotted.

    ax : matplotlib Axes object
        Axes into which the temporal STA is plotted
    """
    if filt.ndim > 1:
        _, temporal_filter = ft.decompose(filt)
    else:
        temporal_filter = filt.copy()
    kwargs['ax'].plot(time, temporal_filter, linestyle='-', linewidth=2, color='LightCoral')


def plot_sta(time, sta):
    """
    Plot a linear filter.

    If the given filter is 1D, it is direclty plotted. If it is 2D, it is
    shown as an image, with space and time as its axes. If the filter is 3D,
    it is decomposed into its spatial and temporal components, each of which 
    is plotted on its own axis.

    Parameters
    ----------
    time : array_like
        A time vector to plot against.

    sta : array_like
        The filter to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure onto which the STA is plotted.

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
def ellipse(filt, sigma=2.0, alpha=0.8, fc='none', ec='black', lw=3, **kwargs):
    """
    Plot an ellipse fitted to the given receptive field.

    Parameters
    ----------
    filt : array_like
        A linear filter whose spatial extent is to be plotted. If this is 2D, it
        is assumed to be the spatial component of the receptive field. If it is
        3D, it is assumed to be a full spatiotemporal receptive field; the spatial
        component is extracted and plotted.

    sigma : float, optional
        Determines the threshold of the ellipse contours. This is the standard
        deviation of a Gaussian fitted to the filter at which the contours are plotted.
        Default is 2.0.

    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque) (Default: 0.8).

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
    fig : matplotlib.figure.Figure
        The figure onto which the ellipse is plotted.
        
    ax : matplotlib.axes.Axes
        The axes onto which the ellipse is plotted.
    """
    fig = kwargs.pop('fig')
    ax = kwargs.pop('ax')

    if filt.ndim == 2:
        spatial_filter = filt.copy()
    elif filt.ndim == 3:
        spatial_filter = ft.decompose(filt)[0]
    else:
        raise ValueError('Linear filter must be 2- or 3-D')

    # get the ellipse parameters
    center, widths, theta = ft.get_ellipse(spatial_filter, sigma=sigma)

    # create the ellipse
    ell = Ellipse(xy=center, width=widths[0], height=widths[1], angle=theta,
                  alpha=alpha, ec=ec, fc=fc, lw=lw, **kwargs)

    ax.add_artist(ell)
    ax.set_xlim(0, spatial_filter.shape[0])
    ax.set_ylim(0, spatial_filter.shape[1])


@plotwrapper
def plot_cells(cells, **kwargs):
    """
    Plot the spatial receptive fields for multiple cells.

    Parameters
    ----------
    cells : list of array_like
        A list of spatiotemporal receptive fields, each of which is a spatiotemporal array.

    ax : matplotlib Axes object, optional
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    Returns
    ------
    fig : matplotlib.figure.Figure
        The figure onto which the ellipses are plotted.

    ax : matplotlib.axes.Axes
        The axes onto which the ellipses are plotted.
    """
    fig = kwargs.pop('fig')
    ax = kwargs.pop('ax')

    # for each cell
    for idx, sta in enumerate(cells):

        # get the spatial profile
        sp = ft.decompose(sta)[0]

        # plot ellipse
        color = cm.Set1(np.random.randint(100))
        fig, ax = ellipse(sp, fc=color, ec=color, lw=2, alpha=0.3, ax=ax)


def play_rates(rates, patches, num_levels=255, time=None, repeat=True, frametime=100):
    """
    Plays a movie representation of the firing rate of a list of cells, by
    coloring a list of patches with a color proportional to the firing rate. This
    is useful, for example, in conjunction with ``plot_cells``, to color the 
    ellipses fitted to a set of receptive fields proportional to the firing rate.

    Parameters
    ----------
    rates : array_like
        An ``(N, T)`` matrix of firing rates. ``N`` is the number of cells, and
        ``T`` gives the firing rate at a each time point.

    patches : list
        A list of ``N`` matplotlib patch elements. The facecolor of these patches is 
        altered according to the rates values.

    Returns
    -------
    anim : matplotlib.animation.Animation
        The object representing the full animation.
    """
    # Validate input
    if rates.ndim == 1:
        rates = rates.reshape(1, -1)
    if isinstance(patches, Ellipse):
        patches = [patches]
    N, T = rates.shape

    # Approximate necessary colormap
    colors = cm.gray(np.arange(num_levels))
    rscale = np.round((num_levels - 1) * (rates - rates.min()) /
                      (rates.max() - rates.min())).astype('int').reshape(N, T)

    # set up
    fig = plt.gcf()
    ax = plt.gca()
    if time is None:
        time = np.arange(T)

    # Animation function (called sequentially)
    def animate(t):
        for i in range(N):
            patches[i].set_facecolor(colors[rscale[i, t]])
        ax.set_title('Time: %0.2f seconds' % (time[t]), fontsize=20)

    # Call the animator
    anim = animation.FuncAnimation(fig, animate, 
            np.arange(T), interval=frametime, repeat=repeat)
    return anim

def anim_to_html(anim):
    """
    Convert an animation into an embedable HTML element.
    
    This converts the animation objects returned by ``play_sta()`` and
    ``play_rates()`` into an HTML tag that can be embedded, for example
    in a Jupyter notebook.

    Paramters
    ---------
    anim : matplotlib.animation.Animation
        The animation object to embed.

    Returns
    -------
    html : IPython.display.HTML
        An HTML object with the encoded video. This can be directly embedded
        into an IPython notebook.

    Raises
    ------
    An ImportError is raised if the IPython modules required to convert the
    animation are not installed.
    """
    from IPython.display import HTML
    return HTML(anim.to_html5_video())
