function e = splitExptStruct(basedir, ename, edate)
%
% FUNCTION e = splitExptStruct(basedir, ename, edate)
%
% Reads the experimental structure from the given experiment and date, and filters
% the various fields, writing all necessary fields to simpler MAT-files.
%
% (c) bnaecker@stanford.edu 2014 
% 18 Jan 2014 - wrote it

%% Make a stim directory
stimdir = fullfile(basedir, 'stim);
if ~exist(stimdir, 'dir')
	mkdir(stimdir);
end

%% Find the appropriate filter file
filtdir = '~/FileCabinet/stanford/baccuslab/projects/datamanagement/filters';
filtfile = fullfile(filtdir, sprintf('%s.txt', ename));
assert(exist(filterfile) ~= 0, ...
	'splitExptStruct:noFilterFile', ...
	sprintf('The filter file for experiment %s does not exist', ename));

%% Open the filter file
fid = fopen(filterfile);

%% Read lines from the file
tmp = textscan(fid, '%s');
filters = tmp{1};
