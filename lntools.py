# lntools.py
#
# Module for linear-nonlinear model data analysis of extracellular recordings
#
# (C) 2014 bnaecker@stanford.edu

## Imports
import scipy as sp
import matplotlib.pyplot as plt

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
	for cell in st:
		hist, tax = sp.histogram(cell, bins = tbins)
		bspikes.append(hist)
	return binspikes, tax

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

		print('working on cell {c} '.format(c = st.index(cell)), end = '')
		
		# Construct list to hold this cell's STE
		thisste = []

		# Number of spikes
		nspikes = len(cell)
		si = 1

		# Loop over spikes
		for spike in cell:

			if sp.mod(si, nspikes / 10) == 0:
				print('.', end = '')

			# Find the block number
			block = sp.where(sp.logical_and(vbl[0, :] < spike, spike <= vbl[-1, :]))[0]

			# Find the frame preceding in this block
			frame = sp.sum(sp.where((vbl[:, block] - spike) < 0, 1, 0)) - 1

			# Add the stimulus of the preceding frames, vectorized
			thisste.append(stim[:, :, frame : frame - nframes : -1].flatten())

		# Append this cell's STE as an nd-array to the full STE
		ste.append(sp.asarray(thisste))
		print('done.')

	# Return the STE
	return ste

def tfsta(tfwn, tfvbl, spikes, upfact = 2, nbands = 3, length = 25):
	'''
	Compute the spike-triggered averages within each tfsens block
	'''

	# Lists to hold the STA and STE
	sta, ste = [], []

	# Compute frames per condition
	fpc = tfvbl.shape[0] / nbands

	# Upsample the white noise stimulus
	upwn = sp.kron(tfwn, sp.ones((upfact,))).flatten()

	# loop over cells
	for cell in spikes:

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

		# Append to the list of all STE's and compute the STA's
		ste.append(stearrays)
		sta.append([sp.mean(st, axis = 0) for st in stearrays])

	# Return the STE and STA lists
	return ste, sta

def plottffilters(filt, nbands = 3, ifi = 0.01, waitframes = 2, upfact = 2):
	'''
	Plot linear filters from the temporal frequency sensitization experiment.

	INPUT:
		filt		- list of filters for each cell. each cell is a list of filters for each band
		nbands		- number of frequency bands
		ifi			- monitor inter-frame interval (for plotting time axis)
		waitframes	- number of frames between monitor flips (also for time axis)

	OUTPUT:
		none
	'''
	# Make a figure
	plt.figure()

	# Determine number of subplots
	ncells = len(filt)
	nplots = [int(sp.sqrt(ncells)), ncells / int(sp.sqrt(ncells))]

	# Make time axis
	flength = len(filt[0][0])
	tax = sp.linspace(0, flength * ifi * waitframes / upfact, flength)

	# Loop over cells
	for ci in sp.arange(ncells):

		# Make a subplot
		plt.subplot(nplots[0], nplots[1], ci + 1)

		# Loop over frequency bands
		for bi in sp.arange(nbands):

			# Plot each filter
			plt.plot(tax, filt[ci][bi], linewidth = 2)

		# Labels etc
		locs, labels = plt.yticks()
		plt.yticks(locs, [''])
		plt.title('cell {c}'.format(c = ci + 1))

		# x-labels only on the bottom row
		if  (ci + 1) >= (nplots[0] - 1) * nplots[1]:
			plt.xlabel('Time before spike (s)')

	# Legend, on the last plot
	plt.legend(('low', 'mid', 'high'))
	return
