# lntools.py
#
# Module for linear-nonlinear model data analysis of extracellular recordings
#
# (C) 2014 bnaecker@stanford.edu

## Imports
import scipy as sp
import matplotlib.pyplot as plt
import seaborn

## Bin spikes
def binspikes(spikes, tmax, binsize=0.01):
	'''
	Bin spike times at the given resolution

	INPUT:
		spikes	- list of spike time arrays for each cell
		tmax	- max time (usually end of experiment or last VBL timestamp)
		binsize	- bin size in msec
	
	OUTPUT:
		bspikes	- list of binned spikes for each cell
		tax		- time axis
	'''
	tbins = sp.arange(0, tmax, binsize)
	bspikes = []
	for cell in spikes:
		hist, tax = sp.histogram(cell, bins = tbins)
		bspikes.append(hist)
	return bspikes, tax

def estimatefr(bspikes, nfiltpts = 7, filtsd = 2):
	'''
	Filter the binned spike times with a Gaussian to obtain an estimate of the firing rate

	INPUT:
		bspikes		- list of binned spike times for each cell
		nfiltpts	- number of pts in the Gaussian filter
		filtsd		- SD of the Gaussian filter

	OUTPUT:
		rates		- list of firing rates for each cell
	'''
	from scipy.signal import gaussian, lfilter
	win = gaussian(nfiltpts, filtsd)
	rates = []
	for cell in bspikes:
		rates.append(lfilter(win, 1, cell))
	return rates

## Spike-triggered ensemble for spatial white noise stimuli
def ste_spatialwn(stim, vbl, st, nframes = 25):
	'''
	Creates the spike-triggered ensemble over the range of VBL timestamps given
	Assumes that the VBL array has been offset appropriately

	INPUT:
		stim	- the stimulus array
		vbl		- array of timestamps for the stimulus
		st		- the spike times for the cell
		nframes - number of frames before the spike to consider as part of STE
	
	OUTPUT:
		ste		- array of stimuli preceding the spike, as an ndarray
	'''

	# Construct list to hold full STE across every cell
	ste = []

	# Loop over cells
	for cell in st:

		# Number of spikes
		nspikes = len(cell)
		si = 0

		# Preallocate ndarray to hold the STE
		thisste = sp.zeros((nspikes, stim.shape[0] * stim.shape[1] * nframes))

		# Notify
		print(' cell {c} of {nc} ({ns} spikes) ...'.format( \
				c = st.index(cell) + 1, nc = len(st), ns = nspikes), end = '', flush = True)

		# Loop over spikes
		for spike in cell:

			# Find the block number
			block = sp.where(sp.logical_and(vbl[0, :] < spike, spike <= vbl[-1, :]))[0]

			# Find the frame preceding in this block
			frame = sp.sum(sp.where((vbl[:, block] - spike) < 0, 1, 0)) - 1

			# Add the stimulus of the preceding frames, vectorized
			if frame < nframes:
				continue
			thisste[si, :] = stim[:, :, frame : frame - nframes : -1].flatten()

			# Increment spike-counter
			si += 1

		# Append this cell's STE as an nd-array to the full STE
		ste.append(thisste)
		print(' done.', flush = True)

	# Return the STE
	return ste

def tfsta(tfwn, tfvbl, spikes, upfact = 2, nbands = 3, length = 25, donorm = False):
	'''
	Compute the spike-triggered averages within each tfsens block

	INPUT:
		tfwn	- the central temporal white noise stimulus
		tfvbl	- array of VBL timestamps during the stimulus
		upfact	- upsample factor. the VBL arrays and stimulus are upsampled to provide better
				  estimates of the linear filters.
		nbands	- number of frequency bands in the stimulus
		length	- number of points in the filter (true value is multiplied by upfact)
		donorm	- flag to normalize each filter as a unit vector, defaults to False

	OUTPUT:
		ste		- the spike-triggered ensemble. a list, with one list of ndarrays for each cell
				  containing the full spike-triggered ensemble for each frequency band
		sta		- the linear filters. a list, with one nbands-by-length ndarray, with the linear
				  filter for each frequency band
	'''

	# Lists to hold the STA and STE
	sta, ste = [], []

	# Compute frames per condition
	fpc = tfvbl.shape[0] / nbands

	# Upsample the white noise stimulus
	upwn = sp.kron(tfwn, sp.ones((upfact,))).flatten()

	# loop over cells
	for cell in spikes:

		# Notify
		print(' cell {c} of {nc} ({ns} spikes) ... '.format( \
				c = spikes.index(cell) + 1, nc = len(spikes), ns = len(cell)), \
				end = '', flush = True)

		# Loop over blocks
		nblocks = tfvbl.shape[-1]
		thisste = [[] for i in sp.arange(0, nbands)]
		for block in sp.arange(0, nblocks):

			# Loop over each frequency band
			for bi in sp.arange(0, nbands):

				# Upsample VBL array
				condstart = bi * fpc
				condend = (bi + 1) * fpc
				vbl = tfvbl[condstart : condend, block]
				upvbl = sp.interp(sp.arange(0, fpc * upfact), 
						sp.arange(0, fpc * upfact, upfact), vbl)

				# Loop over spikes from this block
				spk = cell[sp.where(sp.logical_and(cell > upvbl[0], cell <= upvbl[-1]))]
				for spike in spk:
					# Get immediately preceding frame
					frame = sp.sum(sp.where(upvbl < spike, 1, 0)) - 1 + bi * fpc * upfact
					if frame < length * upfact:
						continue
					# Append the preceding length frames to the STE
					thisste[bi].append(upwn[frame : frame - length * upfact : -1])
	
		# Create nd-array from the spike-triggered stimulus ensemble list
		stearrays = [sp.asarray(st) for st in thisste]
		#raise Exception

		# Append to the list of all STE's
		ste.append(stearrays)

		# STA is the mean of the STE. Join into single ndarray, potentially normalizing each
		if donorm:
			unnormed = [sp.mean(st, axis = 0) for st in stearrays]
			filt = sp.asarray([un / sp.linalg.norm(un) for un in unnormed]).T
		else:
			filt = sp.asarray([sp.mean(st, axis = 0) for st in stearrays]).T

		# Mean-subtract and append to the STA array
		sta.append(filt - (sp.outer(sp.ones((length * upfact,)), sp.mean(filt, axis = 0))))

		# Notify
		print('done.', flush = True)

	# Return the STE and STA lists
	return ste, sta

def tflinearpred(filters, stim, fpc = 1500, upfact = 2):
	'''
	Compute the linear prediction of each TF sensitization filter with the stimulus
	'''
	ncells = len(filters)
	nbands = filters[0].shape[-1]
	lp = [sp.zeros((fpc * upfact,nbands,nbands)) for i in range(ncells)]
	upstim = sp.kron(stim, sp.ones((upfact,))).flatten()
	for ci in range(ncells):
		for bi in range(nbands):
			for bj in range(nbands):
				lp[ci][:, bi, bj] = sp.convolve(upstim[bi * fpc * upfact : (bi + 1) * fpc * upfact], filters[ci][:, bj], 'same')

	return lp

def tfnonlin(filters, lp, ste, nbands = 3, nbins = 30, eps = 1e-3):
	'''
	Compute nonlinearities in each frequency band for the temporal frequency sensitization experiment
	'''
	ncells = len(filters)
	nonlin = [sp.zeros((nbins, nbands, nbands)) for ci in range(ncells)]
	sbins = [sp.zeros((nbins, nbands, nbands)) for ci in range(ncells)]
	stbins = [sp.zeros((nbins, nbands, nbands)) for ci in range(ncells)]
	sdist = [sp.zeros((nbins, nbands, nbands)) for ci in range(ncells)] 
	stdist = [sp.zeros((nbins, nbands, nbands)) for ci in range(ncells)] 
	for ci in range(ncells):
		for bi in range(nbands):
			for bj in range(nbands):

				# Get white noise linear prediction, compute histogram (unconditional)
				pred = lp[ci][:, bi, bj]
				zpred = sp.stats.zscore(pred)
				mu = sp.mean(pred)
				sd = sp.std(pred)
				sdist[ci][:, bi, bj], bins = sp.histogram(pred, nbins, density = True)
				sbins[ci][:, bi, bj] = bins[:-1]

				# Compute spike-triggered distribution
				zst = (ste[ci][bi] - mu) / sd
				stdist[ci][:, bi, bj], bins = sp.histogram(sp.dot(zst, filters[ci][:, bj]), nbins, density = True)
				stbins[ci][:, bi, bj] = bins[:-1]

				# Compute nonlinearity, ratio of distributions
				nonlin[ci][:, bi, bj] = sp.divide(stdist[ci][:, bi, bj], sdist[ci][:, bi, bj] + eps)

	return nonlin, sdist, sbins, stdist, stbins

def plotnl(bins, nl, c = 0, nbands = 3):
	'''
	plot the nonlinearities for the given cell
	'''
	fig = plt.figure()
	fig.canvas.manager.set_window_title('cell ' + str(c))
	for bi in range(nbands):
		for bj in range(nbands):
			plt.subplot(nbands, nbands, bi * 3 + bj + 1)
			plt.plot(bins[c][:, bi, bj], nl[c][:, bi, bj])
			plt.title('stim: ' + str(bi) + ', filt: ' + str(bj))

def plottffilters(filt, nbands = 3, ifi = 0.01, waitframes = 2, upfact = 2):
	'''
	Plot linear filters from the temporal frequency sensitization experiment.

	INPUT:
		filt		- list of filters for each cell. each is an ndarray of filters for each band
		nbands		- number of frequency bands
		ifi			- monitor inter-frame interval (for plotting time axis)
		waitframes	- number of frames between monitor flips (also for time axis)

	OUTPUT:
		none
	'''
	from matplotlib import cm
	colors = cm.Blues(tuple(int(i) for i in \
			sp.linspace(cm.Blues.N / 4, cm.Blues.N - cm.Blues.N / 4, nbands)))
	
	# Make a figure
	plt.figure()

	# Determine number of subplots
	ncells = len(filt)
	nplots = [int(sp.sqrt(ncells)), ncells / int(sp.sqrt(ncells))]

	# Make time axis
	flength = len(filt[0])
	tax = sp.linspace(0, flength * ifi * waitframes / upfact, flength)

	# Loop over cells
	for ci in sp.arange(ncells):

		# Make a subplot
		plt.subplot(nplots[0], nplots[1], ci + 1)

		# Plot all filters for each cell
		for fi in sp.arange(nbands):
			plt.plot(tax, filt[ci][:, fi], linewidth = 2, color = colors[fi, :])

		# Labels etc
		locs, labels = plt.yticks()
		plt.yticks(locs, [''])
		plt.title('cell {c}'.format(c = ci + 1))

		# x-labels only on the bottom row
		if  (ci + 1) > (nplots[0] - 1) * nplots[1]:
			plt.xlabel('Time before spike (s)')

	# Legend, on the last plot
	plt.legend(('low', 'mid', 'high'))
	return

def plotstimfreq(stim, fpc = 1500, ifi = 0.01, waitframes = 2):
	'''
	Plot the frequency spectrum of the TF sensitization stimulus
	'''
	# Compute sample rate, times, axes
	T = ifi * waitframes
	Fs = 1 / T
	L = fpc
	tax = sp.arange(0, L) * T

	# Compute FFT, normalized to the length
	nfft = 2 ** int(sp.ceil(sp.log2(L)))
	ft = sp.fft(stim, n = nfft, axis = 0) / L
	f = Fs / 2 * sp.linspace(0, 1, nfft / 2)
	ps = 2 * sp.absolute(ft[0 : nfft / 2, :])
	
	# Smooth power spectrum a little
	from scipy.signal import lfilter
	npts = 20
	sps = lfilter(sp.ones((npts,)) / npts, 1, ps)

	# Plot
	plt.figure()
	plt.subplot(211)
	plt.plot(tax, stim)
	plt.subplot(212)
	plt.plot(f[:100], sps[:100, :], linewidth = 2)
