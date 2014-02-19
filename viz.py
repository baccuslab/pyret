'''
viz.py

Tools for visualizing data from retinal experiments.

(C) 2014 bnaecker, nirum
'''
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation

def raster(spk, cells=None, trange=None):
	'''
	Usage: fig = raster(spk, cells, trange)
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
		spikes = spk[cell][sp.logical_and(spk[cell] >= trange[0], spk[cell] < trange[1])]
		plt.plot(spikes, (cell + 1) * sp.ones(spikes.shape), color = 'k', marker = '.', linestyle = 'none')

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
	Usage: fig = psth(rates, tax, cells, trange)
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
	plotinds = sp.logical_and(trange[0] <= tax, tax < trange[1])

	# Compute number of subplots
	n = round(sp.sqrt(ncells))
	nplots = (n, sp.ceil(ncells / n))

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

def playsta(sta):
	'''
	Usage: playsta(sta)
	Plays a spatiotemporal spike-triggered average as a movie

	Input
	-----
		
	sta:
		Spike-triggered average array, shaped as (npix, npix, nframes)
	
	Output
	------
	
	None

	'''

	# Initial frame
	im0 = sta[:, :, 0]

	# Set up the figure
	fig = plt.figure()
	ax = plt.axes(xlim=(0, sta.shape[0]), ylim=(0, sta.shape[1]))
	img = plt.imshow(im0)

	# Set up the colors
	maxval = sp.ceil(sp.absolute(sta).max())
	img.set_cmap('gray')
	img.set_interpolation('nearest')
	plt.colorbar()
	
	# Animation initialization function
	def init():
		img.set_data(im0)
		return img

	# Animation function (called sequentially)
	def animate(i):
		ax.set_title('Frame {0:#d}'.format(i + 1))
		img.set_data(sta[:, :, i])
		return img

	# Call the animator
	anim = animation.FuncAnimation(fig, animate, 
			sp.arange(sta.shape[-1]), init_func=init, interval=50, repeat=False)
	plt.show()
	plt.draw()

def plotsta(sta, timeslice=-1):
	'''
	Usage: plotsta(sta, timeslice=-1)
	Plot the given spike-triggered average.

	Input
	-----
	
	sta:
		The spike-triggered average to plot, as an array.

	timeslice:
		Which frame to plot. Default of -1 indicates that the frame with
		the largest absolute deviation from the mean should be plotted.
	
	'''

	# Create the figure
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# Temporal slice to plot
	if timeslice == -1:
		idx = sp.unravel_index(sp.absolute(sta).argmax(), sta.shape)

	# Make the plot plot
	maxval = sp.ceil(sp.absolute(sta).max())
	imgplot = plt.imshow(sta[:, :, timeslice])
	imgplot.set_cmap('RdBu')
	imgplot.set_interpolation('nearest')
	plt.colorbar()
	plt.show()
	plt.draw()
