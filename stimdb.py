'''
stimdb

tools for interacting with the stimulus database

(C) 2014 bnaecker, nirum
'''

# Imports
import os
from scipy.io import loadmat

def dblist(stimtype, verbose=False):
	'''
	usage: stimdb.dblist(type, verbose)
	list stimuli of the given type. verbose controls whether or not to print
	results to the screen.

	input:
		stimtype	- string stimulus type
		verbose		- logical, controls printing of output to screen

	output:
		stimlist	- a dictionary, from stimulus ID tags to dates
	'''

	# Get the stimdb directory
	dbdir = getdir()

	# Find all stimulus IDs of the given type
	stimdir = os.path.join(dbdir, stimtype)
	stimids = os.listdir(stimdir)
	
	# Make a dictionary of the stimulus IDs and the dates on which they were used
	stimlist = {}
	for sid in stimids:
		with open(os.path.join(stimdir, sid, sid + '-dates.txt'), 'rt') as fid:
			stimlist[sid] = [line.rstrip() for line in fid.readlines()]
	
	return stimlist

def idbydate(stimtype, date):
	'''
	usage: stimdb.idbydate(stimtype, date)
	returns the ID of the stimulus of the requested type used on the given date

	input:
		stimtype	- string stimulus type
		date 		- string date

	output:
		stimid		- string stimulus ID
	'''
	# Return the ID (key) whos value (date list) contains the given date
	stimlist = dblist(stimtype);
	return [k for k in stimlist.keys() if date in stimlist[k]][0]

def dbload(stimtype, date):
	'''
	usage: stimdb.load(stimtype, date)
	loads the from the requested date
	
	input:
		stimtype	- string stimulus type
		date 		- string stimulus date
	
	output:
		stim		- dictionary containing the stimulus
	'''
	# Stimulus directory
	dbdir = getdir()

	# Find the ID of the stimulus requested
	stimid = idbydate(stimtype, date)

	# Load the requested stimulus
	matvars = loadmat(os.path.join(dbdir, stimtype, stimid, stimid + '.mat'))

	# Return the non-header stuff
	return {k : matvars[k] for k in matvars.keys() if not k.startswith('__')}

def getdir(local=True):
	'''
	usage: stimdb.getdir(True)
	return the stimulus database directory

	input:
		local	- logical to return the stimdb directory on the local machine,
				  or on the server
	
	output:
		dbdir	- string directory
	'''
	if local:
		 dbdir = '~/FileCabinet/stanford/baccuslab/projects/stimdb'
	else:
		dbdir = '/Volumes/data/Ben/stimdb'
		if not os.path.exists(dbdir):
			dbdir = '~/FileCabinet/stanford/baccuslab/projects/stimdb'

	return os.path.expanduser(dbdir)
