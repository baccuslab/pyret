function e = writeSpikeFiles(datafile, datadir)
%
% FUNCTION e = writeSpikeFiles(datafile, datadir)
%
% The function writeSpikeFiles translates the output of the MATLAB spike-sorting routines
% GroupCW into a text file for each cell found containing the spike times, offset from the
% beginning of the recording.
%
% (c) bnaecker@stanford.edu 2014 
% 18 Jan 2014 - wrote it

%% Return value is 0 if successful
e = 0;

try

%% Load the GroupCW structure array
s = load(datafile);
g = s.g;
clear s;

%% Find channels with spikes
cells = {g.chanclust{~cellfun(@isempty, g.chanclust}};

%% Collect size information
ncells = length(cells);
nfiles = size(cells{1}, 2);

%% Notify
fprintf('making spike time text files\n');

%% Loop over cells, writing each text file
for ci = 1:ncells

	% Notify
	fprintf('cell %d of %d ... ');

	% Open a text file
	fid = fopen(sprintf('%s/c%d.txt', datadir, ci), 'w');

	% Loop over each Igor file
	for fi = 1:nfiles

		% Write the cell number
		fprintf(fid, 'c%d%s\n', ci, 97 + (fi - 1));

		% Write the spike times
		fprintf(fid, '%f\n', [cells{ci}{fi}] ./ g.scanrate);
	end

	% Close the file
	fclose(fid);

	% Notify
	fprintf('done.\n');
end

%% Catch any errors
catch me
	e = me;
end
