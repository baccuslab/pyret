'''
expt.py

tools for finding, loading, listing and saving data from experiments, 
including saving analyzed data

(C) 2014 bnaecker, nirum
'''

# Imports
import os, itertools, stimdb
import scipy as sp
from scipy.io import loadmat

def loadexpt(expttype, date):
	'''
	usage: ex = loadexpt(expttype, date)
	loads the data for the experiment type on the given date

	input:
		expttype	- string experiment type
		date 		- string experiment date

	output:
		vbl			- dict of vbl arrays for each stimulus
		stim		- list of each stimulus
		extra		- anything else in the $DATE-pydat.mat file
	'''

	# Path
	datadir = os.path.join(os.path.expanduser(\
		'~/FileCabinet/stanford/baccuslab/projects/'), expttype, 'data', date)

	# Load the MAT-file containing Python data
	pydat = loadmat(os.path.join(datadir, date + '-pydat.mat'))

	# Pull out the vbl arrays, anything else is "extra"
	vbl = {k : pydat[k] for k in pydat.keys() if k.endswith('vbl')}
	extra = {k : pydat[k] for k in pydat.keys() \
			if not k.startswith('__') and not k.endswith('vbl')}

	# Find the stimulus references
	with open(os.path.join(datadir, date + '-stimref.txt'), 'rt') as fid:
		lines = [line.rstrip().split('\t') for line in fid.readlines()]
		stimref = {line[0] : line[1] for line in lines}

	# Load them
	stim = dict(itertools.chain(*[d.items() for d in \
			[stimdb.dbload(t, date) for t in stimref.keys()]]))

	# Return 
	return vbl, stim, extra

def listexpts(expttype):
	'''
	usage: l = listexpts(expttype)
	lists all experiments of the given type

	input:
		expttype	- string experiment type

	output:
		l			- list of dates for this experiment
	'''
	return os.listdir(os.path.join(os.path.expanduser(\
			'~/FileCabinet/stanford/baccuslab/projects', expttype, 'data')))
