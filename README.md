# pyret
![pyret logo](https://raw.githubusercontent.com/baccuslab/pyret/dev/logo.png?token=904854__eyJzY29wZSI6IlJhd0Jsb2I6YmFjY3VzbGFiL3B5cmV0L2Rldi9sb2dvLnBuZyIsImV4cGlyZXMiOjE0MDg1Nzg2OTZ9--973fe76a2dce2d2ed5a65b679865053ca50197ff)

### A Python package for analyzing retinal data
Benjamin Naecker, Niru Maheswaranthan, Pablo Jadzinsky

Brief description
-----------------

The pyret package provides a set of tools useful in the analysis of retina experiments
in the Baccus lab. It contains routines for interacting with raw Igor binary files
from the recording computer, manipulating spike trains, performing basic spike-triggered
analyses, and visualization tools for all these.

The package describes a Cell class, which contains data and functionality associated 
with individual cells recorded during an experiment. The class is described in detail 
below.

Demo
----
For a demo of how to do analysis using `pyret`, check out the html file and corresponding ipython notebook in the `demo/` folder.

Testing
-------
Testing is done via the [nose](https://nose.readthedocs.org/en/latest/) package. Once installed (e.g. with `pip install nose`) then simply run `nosetests -v` at the top level directory to run the tests. Test functions are located in the `tests/` folder.

Submodule overview
------------------

[**binary**](#binary)		-- reading binary recording files, useful for checking the photodiode

[**spiketools**](#spk)		-- basic manipulation of spike-times, including binning & smoothing

[**filtertools**](#sta)		-- computing components of simple linear-nonlinear models, including linear filters and nonlinearities

[**stimulustools**](#stim)	-- tools for manipulating stimuli, e.g., upsampling.

[**visualization**](#viz)	-- visualization methods

Classes overview
----------------

[**Cell**](#cell)			-- individual cell

Submodules in detail
--------------------

<h3 id="binary">binary</h3>
<hr>
Tools for reading Igor binary files, particularly for interacting with the photodiode.

`readbinhdr(fname)`
Read the header from the file `fname`

`readbin(fname, chan=0)`
Read the channel `chan` from the file `fname`

<h3 id="spk">spiketools</h3>
<hr>
Tools for manipulating spike-time arrays.

`binspikes(spk, binsize=0.01)`
Bin spike times at the given resolution

`estfr(bspk, npts=9, sd=2)`
Estimate firing rate by smoothing binned spikes

<h3 id="sta">filtertools</h3>
<hr>
Tools for computing linear filters of various kinds.

`getste(time, stimulus, spikes, length)`
Construct the spike-triggered stimulus ensemble

`getsta(time, stimulus, spikes, length)`
Compute the spike-triggered average

`getstc(time, stimulus, spikes, length)`
Compute the spike-triggered covariance

`lowranksta(sta, k=10)`
Compute a rank-k approximation to the given spatiotemporal STA

`decompose(sta)`
Decompose the given spatiotemporal STA into a spatial and temporal kernel.

<h3 id="stim">stimulustools</h3>
<hr>
Stimulus tools.

`upsamplestim(time, stim, upfact)`
Upsample the stimulus by the given factor.

`downsamplestim(time, stim, upfact)`
Downsample the stimulus by the given factor.

`slicestim(stim, history, locations=None)`
Take slices of length `history` from the given stimulus, optionally
specifying the temporal locations of the slices.

<h3 id="viz">visualization</h3>
<hr>
Visualization tools.

`raster(spk, trange=None)`
Plot spike raster over the given time range

`psth(spk, trange=None)`
Plot psth over the given time range

`playsta(sta, repeat=True, frametime=100)`
Play a spatio-temporal STA as a movie

`plotsta(time, sta, timeslice=None)`
Plot the spatial and temporal kernels of a spatiotemporal STA

`temporal(time, temporalfilter, ax=None)`
Plot the given temporal filter

`spatial(spatialfilter, ax=None)`
Plot the given spatial filter

<h2 id="cell">Cell class</h2>
<h4>Constructor</h4>
`c = cell.Cell(spk=None)`
Construct a Cell object, optionally giving the spike times for the cell

<h4>Data</h4>

`celltype`	- A string defining the cell-type (e.g., 'on')

`notes`		- A string with user-defined notes

`uid`		- A string/date unique ID for the cell. *Not yet implemented.*

`spk` 		- Array of spike times

`sta` 		- Spike-triggered average for the cell

`filtax`	- Time axis for the STA/STE

`ste` 		- Spike-triggered stimulus ensemble

`nonlin`	- Nonlinearity for the cell. *Not yet implemented.*

`nonlinax`	- Axis for the nonlinearity. *Not yet implemented*

<h4>Functions</h4>

`settype`, `setnotes`, `setuid`
Setter methods for the corresponding attributes

`getsta` 	- Compute the spike-triggered average

`getste` 	- Compute the spike-triggered average

`getstc`	- Compute the spike-triggered covariance. *Not yet implemented.*

`getnonlin`	- Compute the cell's nonlinarity. *Not yet implemented.*

`plot` 		- General plotting function for showing the linear filters in various forms

