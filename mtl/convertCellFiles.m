function e = convertCellFiles(exptName, exptDate)
%
% FUNCTION e = convertCellFiles(exptName, exptDate)
%
% The function convertCellFiles reads the output of the Matlab spike-sorting
% software GroupCW, and creates text files for each cell with that cells spike
% times.
%
% INPUT: 
%	'exptName'		- the name of the experiment
%	'exptDate'  	- the date of the experiment
%
% OUTPUT: the output is simply an error struct if anything went wrong, or 0 if
% the conversion was successful.
%
% (c) bnaecker@stanford.edu 2013 
% 13 Feb 2013 - wrote it
% 10 May 2013 - removed varagin, simplified to just expt and date

try

%% format date for the server
serverDate = [exptDate(3:4) exptDate(1:2) exptDate(5:6)];

% base directory
baseDir = ['~/FileCabinet/stanford/baccuslab/projects/' exptName ...
	'/Data/' exptDate];
datafile = fullfile('/Volumes/data/Ben/', exptName, serverDate, [serverDate 'a.mat']);
matfile = fullfile('/Volumes/data/Ben/', exptName, serverDate, '201*');

% check if the server is connected
if exist(datafile, 'file') == 0
	% try other date format
	datafile = fullfile('/Volumes/data/Ben/', exptName, exptDate, [serverDate 'a.mat']);
	matfile = fullfile('/Volumes/data/Ben/', exptName, exptDate, '201*');
	if exist(datafile, 'file') == 0
		error('convertCellFiles:badDataFile', ...
			sprintf('Could not find file %s. The server may not be mounted', ...
			datafile));
	end
end
s = load(datafile);
g = s.g;
clear s;

% make data directory
if ~exist(baseDir, 'dir')
	mkdir(baseDir);
end

%% find channels with spikes
cells = {g.chanclust{~cellfun(@isempty, g.chanclust)}};

%% make spike times file for each cell
nCells = length(cells);
nFiles = size(cells{1}, 2);

% notify
fprintf('making spike time text files ... ');

for ci = 1:nCells
	% open text file
	fid = fopen([baseDir '/c' num2str(ci) '.txt'], 'w');
	
	% loop over experiment files
	for fi = 1:nFiles

		% write cell number
		fprintf(fid, 'c%d%s\n', ci, 97 + (fi - 1));

		% write spike times
		fprintf(fid, '%f\n', [cells{ci}{fi}] ./ g.scanrate);
	end
	% close file
	fclose(fid);
end
% notify
fprintf('done.\n');

%% also copy experimental structure
fprintf('copying experimental structure from server ... ');
system(['rsync -avz ' matfile ' ' baseDir ]);
fprintf('done.\n');

% got here? no error
e = 0;
catch me
	e = me;
end
