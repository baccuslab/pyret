# dtools.py
#
# Basic data manipulation tools for Baccus lab experimental data.
#
# (c) 2014 bnaecker@stanford.edu

## Imports
import scipy as sp				# For lots of stuff
from scipy.io import loadmat	# Loading MATLAB files
from os import path, listdir	# Finding files
import re						# Matching filenames mostly

## Load data about an experiment
def loadexp(name, date):
	'''
	Loads the stimulus and VBL timestamp arrays from the given experiment.

	INPUT:
		name	- string, experiment type
		date 	- string, experiment date
	
	OUTPUT:
		VBL arrays and stimuli for the given experiment, as a dictionary
	'''
	# Path information
	datadir = path.join(\
			path.expanduser('~/FileCabinet/stanford/baccuslab/projects/'), \
			name, 'data', date)

	# Load MAT-file for Python
	pydat = loadmat(path.join(datadir, 'pydat.mat'));

	# Return the useful keys and arrays
	return {k: pydat.get(k) for k in pydat.keys() if not k.startswith('__')}

## Load the spikes times of an experiment
def loadspikes(name, date):
	'''
	Loads the spike times for each cell from the given experiment.

	INPUT:
		name	- string, experiment name
		date 	- string, experiment date

	OUTPUT:
		spikes	- list of spike times for each cell
	'''
	# Path information
	datadir = path.join(\
			path.expanduser('~/FileCabinet/stanford/baccuslab/projects/'), \
			name, 'data', date)

	# Find the cell files
	cfiles = [path.join(datadir, f) for f in listdir(datadir) if re.match('c[0-9]*.txt', f)]

	# Return a list of all the cells
	return [sp.loadtxt(cf) for cf in cfiles]

def readigorbin(fname, chan = 0):
	'''
	Read an Igor binary file.
	
	INPUT:
		fname	- the full path of the file to read
		chan	- which channel to read, defaults to 0, the photodiode
	
	OUTPUT:
		hdr		- the file header
		dat		- the channel data, as a numpy ndarray
	'''
	if not path.exists(fname):
		print('Requested bin file {f} does not exist'.format(f = fname))
		raise FileNotFoundError

	# Context manager to read the file (automagically closes when done)
	with open(fname, 'rb') as fid:

		# Rewind the file
		fid.seek(0)

		# Reade the header
		hdr = readigorhdr(fid)

		# Check that the requested channel is OK
		if chan not in hdr['channel']:
			print('Requested channel {c} is not in the file'.format(c = chan))
			raise IndexError

		# A bit more setup
		nblocks = int(hdr['nsamples'] / hdr['blksize'])
		skip = hdr['nchannels'] * hdr['blksize']
		chanoffset = chan * hdr['blksize']

		# Setup ndarray to return the values
		dat = sp.zeros((hdr['nsamples'] / hdr['nchannels'],))

		# Read the requested channel, a block at a time
		fid.seek(hdr['hdrsize'])
		for block in range(nblocks):
			pos = hdr['hdrsize'] + block * skip + chanoffset
			fid.seek(pos)
			dat[block * hdr['blksize'] : (block + 1) * hdr['blksize']] = \
				sp.fromfile(fid, dtype = sp.dtype('>i2'), count = hdr['blksize'])

		# Scale and offset
		dat *= hdr['gain']
		dat += hdr['offset']

	# Return stuff
	return hdr, dat

def readigorhdr(fid):
	'''
	Read the header of an Igor binary file.

	INPUT:
		fid		- file pointer, previously returned with open('file.bin', 'rb')
	
	OUTPUT:
		hdr		- dict of the header information
	'''

	# Define some data types
	uint 	= sp.dtype('>u4') 	# Unsigned integer, 32-bit
	short 	= sp.dtype('>i2') 	# Signed 16-bit integer
	flt 	= sp.dtype('>f4') 	# Float, 32-bit
	uchar 	= sp.dtype('>B') 	# Unsigned char

	# Read the header
	hdr = {}
	hdr['hdrsize'] 		= sp.fromfile(fid, dtype = uint, count = 1) 	# size of header (bytes)
	hdr['type']			= sp.fromfile(fid, dtype = short, count = 1)	# not sure
	hdr['version']		= sp.fromfile(fid, dtype = short, count = 1) 	# not sure
	hdr['nsamples']		= sp.fromfile(fid, dtype = uint, count = 1) 	# samples in file
	hdr['nchannels']	= sp.fromfile(fid, dtype = uint, count = 1) 	# number of channels
	hdr['channel'] 		= sp.fromfile(fid, dtype = short, count = hdr['nchannels']) 	# channels
	hdr['fs']			= sp.fromfile(fid, dtype = flt, count = 1)		# sample rate
	hdr['blksize'] 		= sp.fromfile(fid, dtype = uint, count = 1)		# sz of data blocks
	hdr['gain']			= sp.fromfile(fid, dtype = flt, count = 1)		# amplifier gain
	hdr['offset']		= sp.fromfile(fid, dtype = flt, count = 1)		# amplifier offset
	hdr['datesz']		= sp.fromfile(fid, dtype = uint, count = 1)		# size of date string
	tmpdate				= sp.fromfile(fid, dtype = uchar, count = hdr['datesz'])		# date
	hdr['timesz']		= sp.fromfile(fid, dtype = uint, count = 1)		# size of time string
	tmptime				= sp.fromfile(fid, dtype = uchar, count = hdr['timesz'])		# time
	hdr['roomsz']		= sp.fromfile(fid, dtype = uint, count = 1)		# size of room string
	tmproom				= sp.fromfile(fid, dtype = uchar, count = hdr['roomsz'])		# room

	# Convert the strings to actual strings
	hdr['date'] = ''.join([chr(i) for i in tmpdate])
	hdr['time'] = ''.join([chr(i) for i in tmptime])
	hdr['room'] = ''.join([chr(i) for i in tmproom])

	# Return the header
	return hdr

