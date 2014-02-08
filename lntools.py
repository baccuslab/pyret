'''
lntools.py

tools for basic linear-nonlinear model components

(C) 2014 bnaecker, nirum
'''

# Imports
import scipy as sp
import matplotlib.pyplot as plt
import seaborn

def getste(stim, vbl, spk, nframes=25, reshape=False):
	'''
	usage: ste = getste(stim, vbl, spk, nframes)
	construct the spike-triggered ensemble

	input:
		stim	- the stimulus array. can be 1-, 2-, or 3-dimensional, corresponding
				  to temporal, lines, or spatial white noise stimuli
		vbl		- the vbl timestamp array
		spk		- list of spike-time arrays for each cell
		nframes	- number of frames to consider for the ste
		reshape - reshape the STE to return it as an array of (stimdim1, ..., stimdimN, nframes)
				  rather than the default prod(stimsize) * nframes

	output:
		ste		- the spike-triggered ensemble, same shape as stim
	'''
	if sp.ndim(stim) == 3:
		ste = _ste_spatialwn(stim, vbl, spk, nframes, reshape)
	elif sp.ndim(stim) == 2:
		ste = _ste_lineswn(stim, vbl, spk, nframes, reshape)
	elif sp.ndim(stim) == 1:
		ste = _ste_temporalwn(stim, vbl, spk, nframes, reshape)

	return ste

def _ste_spatialwn(stim, vbl, spk, nframes, reshape):
	'''
	spike-triggered ensemble for spatial white noise stimuli. not called directly, 
	only through lntools.getste
	'''
	# Return list
	ncells = len(spk)
	ste = []

	# Loop over cells
	for cell in spk:

		# Capture white noise spikes only
		wnspikes = cell[sp.logical_and(cell > vbl[0], cell <= vbl[-1])]
		nspikes = wnspikes.size

		# Prealloc array to hold STE
		wnste = sp.zeros((nspikes, stim.shape[0] * stim.shape[1] * nframes))

		# Notify
		print(' cell {c} of {nc} ({ns} spikes) ...'.format(\
				c = spk.index(cell) + 1, nc = ncells, ns = nspikes), \
				end = '', flush = True)

		# Loop over spikes
		si = 0
		for spike in wnspikes:

			# Find VBL array block
			block = sp.logical_and(vbl[0, :] < spike, spike <= vbl[-1, :])

			# Add immediately preceding nframes frames
			frame = sp.sum(vbl[:, block] < spike) - 1
			if frame < nframes:
				continue
			wnste[si, :] = stim[:, :, frame : frame - nframes : -1].flatten()
			si += 1

		# Append this ste to the list
		ste.append(wnste)
		print(' done.', flush = True)

	if reshape:
		return [s.reshape(s.shape[0], stim.shape[0], stim.shape[1], nframes)\
				for s in ste]
	return ste
			
def _ste_lineswn(stim, vbl, spk, nframes):
	'''
	spike-triggered ensemble for lines white noise stimuli. not called directly, 
	only through lntools.getste
	'''
	raise NotImplemented

def _ste_temporalwn(stim, vbl, spk, nframes):
	'''
	spike-triggered ensemble for temporal white noise stimuli. not called directly, 
	only through lntools.getste
	'''
	raise NotImplemented

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
