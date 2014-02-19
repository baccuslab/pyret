# pyret
## A Python package for messing with retinal data
(C) 2014 Benjamin Naecker, Niru Maheshwaranthan

Brief description
-----------------

The pyret package provides a set of tools useful in the analysis of retina experiments
in the Baccus lab. It contains routines for interacting with raw Igor binary files
from the recording computer, manipulating spike trains, performing basic spike-triggered
analyses, and visualization tools for all these.

The package describes a Cell class, which contains data and functionality associated 
with individual cells recorded during an experiment. The class is described in detail 
below.

Submodule overview
------------------

[**rbin**](#binary)		-- reading binary recording files, useful for checking the photodiode

[**spk**](#spk)			-- basic manipulation of spike-times, including binning & smoothing

[**sta**](#sta)			-- computing components of simple linear-nonlinear models, including linear filters and nonlinearities

[**viz**](#viz)			-- visualization methods

Classes overview
----------------

[**Cell**](#cell)			-- individual cell

Submodules in detail
--------------------

<h3 id="binary">rbin</h3>
<hr>
Tools for reading Igor binary files, particularly for interacting with the photodiode.

`readbinhdr(fname)`
read the header from the file `fname`

`readbin(fname, chan=0)`
read the channel `chan` from the file `fname`

<h3 id="spk">spk</h3>
<hr>
Tools for manipulating spike-time arrays.

`binspikes(spk, binsize=0.01)`
bin spike times at the given resolution

`estimatefr(bspk, npts=9, sd=2)`
estimate firing rate by smoothing binned spikes

<h3 id="sta">sta</h3>
<hr>
Tools for performing spike-triggered average analyses

`getste(stim, vbl, spk, nframes=25)`
find the spike-triggered ensemble

`sta(stim, vbl, spk, nframes=25)`
compute the spike-triggered average

`stc(stim, vbl, spk, nframes=25)`
compute the spike-triggered covariance

`nonlin(stim, sta, nbins=30)`
compute nonlinearities

<h3 id="viz">viz</h3>
<hr>
Visualization tools.

`raster(spk, trange=None)`
plot spike raster over the given time range

`psth(spk, trange=None)`
plot psth over the given time range

`playsta(sta, trange=None)`
play a spatio-temporal STA as a movie

<h2 id="cell">Cell class</h2>
<h4>Data</h4>

`spk` - array of spike times

`sta` - array containing the spike-triggered average

`ste` - array containing the spike-triggered ensemble

<h4>Functions</h4>

`getsta` - compute the spike-triggered average

`getste` - compute the spike-triggered average

`plot` - plot the spike-triggered average

`psth` - plot a PSTH
