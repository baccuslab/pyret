'''
visualizations.py

Tools for visualizing data from retinal experiments.

(C) 2014 bnaecker, nirum
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import filtertools as ft
from matplotlib import animation

def raster(spk, cells=None, trange=None):
	'''
	
	Plot a raster of spike times over the given time

	Input
	-----

	spk:
		List of spike times for each cell

	cells:
		List of which cells to plot (None == all)

	trange:
		2-elem tuple giving time range (None = (min(spk), max(spk)))

	Output
	------

	fig:
		Matplotlib handle of the figure

	'''

	# Parse input
	if cells is None:
		cells = range(0, len(spk))
	else:
		cells = [c for c in cells if 0 < c <= len(spk)]

	if trange is None:
		trange = (min([s.min() for s in spk]), max([s.max() for s in spk]))
	else:
		trange = (max(trange[0], 0), min(trange[1], max([s.max() for s in spk])))

	# Plot rasters for each cell
	fig = plt.figure()
	ncells = len(cells)
	for cell in range(ncells):
		spikes = spk[cell][np.logical_and(spk[cell] >= trange[0], spk[cell] < trange[1])]
		plt.plot(spikes, (cell + 1) * np.ones(spikes.shape), color = 'k', marker = '.', linestyle = 'none')

	# Labels etc
	plt.title('spike rasters', fontdict={'fontsize':24})
	plt.xlabel('time (s)', fontdict={'fontsize':20})
	plt.ylabel('cell #', fontdict={'fontsize':20})
	plt.ylim(ymin = 0, ymax=ncells + 1)
	plt.show()
	plt.draw()

	return fig

def psth(rates, tax, cells=None, trange=None):
	'''
	
	Plot psths for the given cells over the given time

	Input
	-----

	rates:
		List of firing rates for each cell

	tax:
		Time axis for firing rates

	cells:
		List of which cells to plot (None == all)

	trange:
		2-elem tuple giving time range (None == (min(tax), max(tax)))

	Output
	------
		fig	- Matplotlib figure handle

	'''

	# Parse input
	if cells is None:
			cells = range(0, len(rates))
	else:
			cells = [c for c in cells if 0 < c <= len(rates)]
	ncells = len(cells)

	if trange is None:
			trange = (tax.min(), tax.max())
	else:
			trange = (max(trange[0], 0), min(trange[1], tax.max()))

	# Compute plot indices
	plotinds = np.logical_and(trange[0] <= tax, tax < trange[1])

	# Compute number of subplots
	n = round(np.sqrt(ncells))
	nplots = (n, np.ceil(ncells / n))

	# Plot psths for each cell
	fig = plt.figure()
	for cell in range(ncells):
		plt.subplot(nplots[0], nplots[1], cell + 1)
		plt.plot(tax[plotinds], rates[cell][plotinds], color = 'k', marker = None, linestyle = '-')

		# Labels etc
		plt.title('cell {c} psth'.format(c = cell + 1), fontdict={'fontsize': 24})
		plt.xlabel('time (s)', fontdict={'fontsize':20})
	plt.show()
	plt.draw()

	return fig

def playsta(sta, repeat=True, frametime=100):
	'''
	
	Plays a spatiotemporal spike-triggered average as a movie

	Input
	-----

	sta:
		Spike-triggered average array, shaped as (npix, npix, nframes)

    repeat [optional, default=True]:
        Whether or not to repeat the animation

    frametime [optional, default=100]:
        Length of time each frame is displayed for (in milliseconds)

	Output
	------

	None

	'''

	# Initial frame
	initialFrame = sta[:, :, 0]

	# Set up the figure
	fig = plt.figure()
	ax = plt.axes(xlim=(0, sta.shape[0]), ylim=(0, sta.shape[1]))
	img = plt.imshow(initialFrame)

	# Set up the colors
	maxval = np.ceil(np.absolute(sta).max())
	img.set_cmap('gray')
	img.set_interpolation('nearest')
	plt.colorbar()

	# Animation initialization function
	def init():
		img.set_data(initialFrame)
		return img

	# Animation function (called sequentially)
	def animate(i):
		ax.set_title('Frame {0:#d}'.format(i + 1))
		img.set_data(sta[:, :, i])
		return img

	# Call the animator
	anim = animation.FuncAnimation(fig, animate,
			np.arange(sta.shape[-1]), init_func=init, interval=frametime, repeat=False)
	plt.show()
	plt.draw()

def spatial(spatialFrame, ax=None):
    '''
	
	Plot a spatial filter on a given axes

	Input
	-----

	spatialFrame:
		The frame to plot, as an (n x n) matrix.

    ax [optional]:
        the axes on which to plot the data; defaults to creating a new figure

    Output
    ------

    axes handle

    '''

    if not ax:
        ax = plt.figure().add_subplot(111)

    img = ax.imshow(spatialFrame)
    img.set_cmap('RdBu')
    img.set_interpolation('nearest')
    plt.colorbar()
    plt.show()
    plt.draw()

    return ax

def temporal(time, temporalFilter, ax=None):
    '''
	
	Plot a temporal filter on a given axes

    Input
    -----

    time:
        a time vector to plot against

    temporalFilter:
        the temporal filter to plot, has the same dimensions as time

    ax [optional]:
        the axes on which to plot the data; defaults to creating a new figure

    Output
    ------

    axes handle

    '''

    if not ax:
        ax = plt.figure().add_subplot(111)

    ax.plot(time, temporalFilter, linestyle='-', linewidth=2, color='LightCoral')
    plt.show()
    plt.draw()

    return ax

def plotsta(time, sta, timeSlice=None):
    '''
	
	Plot a spatial and temporal filter

    Input
    -----

    time:
        a time vector to plot against

    sta:
        the filter to plot

    timeslice [optional]:
        the index of the spatial slice to plot

    Output
    ------

    axes handle

    '''

    # create the figure
    fig = plt.figure()

    # decompose
    spatialProfile, temporalFilter = ft.decompose(sta)

    # plot spatial profile
    axspatial = spatial(spatialProfile, fig.add_subplot(121))

    # plot temporal profile
    axtemporal = temporal(time, temporalFilter, fig.add_subplot(122))

    # return handles
    return fig, (axspatial, axtemporal)

def ellipse(ell, ax=None):
	'''
	
	Plot the given ellipse, fit to the spatial receptive field of a cell

	Input
	-----

	ell:
		A matplotlib.patches.Ellipse object

	ax [optional]:
		The axes onto which the ellipse should be plotted. Defaults to a new figure

	Output
	------

	ax:
		The axes onto which the ellipse is plotted
	
	'''

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
