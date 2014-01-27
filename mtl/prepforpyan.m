function e = prepforpyan(ename, edate)
%
% FUNCTION e = prepforpyan(ename, edate)
%
% The function prepforpyan prepares various data structures for further analysis in 
% my Python pipeline. Because the Baccus Lab uses MATLAB/PTB for stimulus presentation,
% and the spike-sorting software is also currently written in MATLAB, parsing the output
% data strutures directly in Python is very cumbersome. This function finds, parses, and
% collects the various MATLAB files used in experiments, and saves the information
% necessary for futher Python analysis into easy-to-access MAT-files and text files.
%
% INPUT:
%	ename	-	string name of the experiment
%	edate	- 	string date of the experiment
%
% (c) bnaecker@stanford.edu 2014 
% 18 Jan 2014 - wrote it

%% Check inputs
if ~(isstr(ename) && isstr(edate))
	error('prepforpyan:nonStringInput', ...
		'The experiment name and date must be strings');
end

%% Base directory information
sdate = [edate(3:4) edate(1:2) edate(5:end)];
basedir = fullfile('~/FileCabinet/stanford/baccuslab/projects', ename, 'data', edate);
datafile = fullfile('/Volumes/data/Ben/', ename, edate, sprintf('%s.mat', sdate));

%% Check if the server is connected
if exist(datafile, 'file') == 0
	error('prepforpyan:serverNotMounted', ...
		sprintf('Could not find the requested data file %s. The server may not be mounted.', ...
		datafile));
end

%% Make the analysis data directory, if it does not exist already
if ~exist(basedir, 'dir')
	mkdir(basedir);
end

%% Notify
fprintf('\n-----------------------------------------\n');
fprintf('prepping data from %s experiment on %s\n', ename, edate);
fprintf('------------------------------------------\n');

%% Write the spike-time files for each cell
e = writeSpikeFiles(datafile, basedir);

%% Rethrow any error
if isa(e, 'MException');
	rethrow(e);
end

%% Parse the experimental structure
e = splitExptStruct(basedir, ename, edate);
