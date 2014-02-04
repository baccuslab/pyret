# dtools.py
#
# Basic data manipulation tools for Baccus lab experimental data.
#
# (c) 2014 bnaecker@stanford.edu

## Imports
from scipy import loadtxt, zeros, fromfile, dtype	# Loading, preallocating data
from scipy.io import loadmat						# Loading MATLAB files
from os import listdir								# Finding files
from os.path import join, expanduser, exists		# Pathnames
from re import match								# For matching filenames

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
	datadir = join(\
			expanduser('~/FileCabinet/stanford/baccuslab/projects/'), \
			name, 'data', date)

	# Load MAT-file for Python
	pydat = loadmat(join(datadir, 'pydat.mat'));

	# Return the useful keys and arrays
	return {k: pydat.get(k) for k in pydat.keys() if not k.startswith('__')}

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
	datadir = join(\
			expanduser('~/FileCabinet/stanford/baccuslab/projects/'), \
			name, 'data', date)

	# Find the cell files
	cfiles = [join(datadir, f) for f in listdir(datadir) if match('c[0-9]*.txt', f)]

	# Return a list of all the cells
	return [loadtxt(cf) for cf in cfiles]

def readigorbin(fname, chan = 0):
	'''
	Read an Igor binary file.
	
	INPUT:
		fname	- the full path of the file to read
		chan	- which channel to read, defaults to 0, the photodiode
	
	OUTPUT:
		hdr		- the file header, a Python dict
		dat		- the channel data, as a numpy ndarray
	'''

	# Some read constants
	uint = dtype('>i2')		# 16-bit unsigned integers

	# Check file exists
	if not exists(fname):
		print('Requested bin file {f} does not exist'.format(f = fname))
		raise FileNotFoundError

	# Context manager to read the file (automagically closes when done)
	with open(fname, 'rb') as fid:

		# Rewind the file
		fid.seek(0)

		# Read the header
		hdr = readigorhdr(fid)

		# Check that the requested channel is in the recording
		if chan not in hdr['channel']:
			print('Requested channel {c} is not in the file'.format(c = chan))
			raise IndexError

		# Compute number of blocks and some offsets
		nblocks = int(hdr['nsamples'] / hdr['blksize']) * uint.itemsize
		skip = hdr['nchannels'] * hdr['blksize'] * uint.itemsize
		chanoffset = chan * hdr['blksize'] * uint.itemsize

		# Preallocate ndarray to return the values
		dat = zeros((hdr['nsamples'],))

		# Read the requested channel, a block at a time
		fid.seek(hdr['hdrsize'])
		for block in range(nblocks):
			# Compute start of the block and set the file position
			pos = hdr['hdrsize'] + block * skip + chanoffset
			fid.seek(pos)

			# Read the data
			dat[block * hdr['blksize'] : (block + 1) * hdr['blksize']] = \
				fromfile(fid, dtype = uint, count = hdr['blksize'])

	# Scale and offset
	dat *= hdr['gain']
	dat += hdr['offset']

	# Return the header and the actual data
	return hdr, dat

def readigorhdr(fid):
	'''
	Read the header of an Igor binary file.

	INPUT:
		fid		- file pointer, previously returned with open('file.bin', 'rb')
	
	OUTPUT:
		hdr		- dict of the header information
	'''

	# Define datatypes to be read in
	uint 	= dtype('>u4') 	# Unsigned integer, 32-bit
	short 	= dtype('>i2') 	# Signed 16-bit integer
	flt 	= dtype('>f4') 	# Float, 32-bit
	uchar 	= dtype('>B') 	# Unsigned char

	# Read the header
	hdr = {}
	hdr['hdrsize'] 		= fromfile(fid, dtype = uint, count = 1) 				# size of header (bytes)
	hdr['type']			= fromfile(fid, dtype = short, count = 1)				# not sure
	hdr['version']		= fromfile(fid, dtype = short, count = 1) 				# not sure
	hdr['nsamples']		= fromfile(fid, dtype = uint, count = 1) 				# samples in file
	hdr['nchannels']	= fromfile(fid, dtype = uint, count = 1) 				# number of channels
	hdr['channel'] 		= fromfile(fid, dtype = short, count = hdr['nchannels'])# channels
	hdr['fs']			= fromfile(fid, dtype = flt, count = 1)					# sample rate
	hdr['blksize'] 		= fromfile(fid, dtype = uint, count = 1)				# sz of data blocks
	hdr['gain']			= fromfile(fid, dtype = flt, count = 1)					# amplifier gain
	hdr['offset']		= fromfile(fid, dtype = flt, count = 1)					# amplifier offset
	hdr['datesz']		= fromfile(fid, dtype = uint, count = 1)				# size of date string
	tmpdate				= fromfile(fid, dtype = uchar, count = hdr['datesz'])	# date
	hdr['timesz']		= fromfile(fid, dtype = uint, count = 1)				# size of time string
	tmptime				= fromfile(fid, dtype = uchar, count = hdr['timesz'])	# time
	hdr['roomsz']		= fromfile(fid, dtype = uint, count = 1)				# size of room string
	tmproom				= fromfile(fid, dtype = uchar, count = hdr['roomsz'])	# room

	# Convert the date, time and room to strings
	hdr['date'] = ''.join([chr(i) for i in tmpdate])
	hdr['time'] = ''.join([chr(i) for i in tmptime])
	hdr['room'] = ''.join([chr(i) for i in tmproom])

	# Return the header
	return hdr
