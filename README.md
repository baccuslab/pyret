# pyret
## A Python package for messing with retinal data
(C) 2014 Benjamin Naecker, Niru Maheshwaranthan

Brief description
-----------------

The pyret package provides a set of tools useful in the analysis of retina experiments
in the Baccus lab. It contains routines for interacting with raw Igor binary files
from the recording computer, manipulating spike trains, performing basic spike-triggered
analyses, a few retina simulation tools, and visualization tools for all these.

Submodule overview
------------------

[**binary**](#binary)	-- reading binary recording files, useful for checking the photodiode

[**spk**](#spk)			-- basic manipulation of spike-times, including binning & smoothing

[**sta**](#sta)			-- computing components of simple linear-nonlinear models, including linear filters and nonlinearities

[**retsim**](#retsim)	-- tools for simulating the retina (LN, LN-LN, LNIAF, LNK, LNKS models)

[**viz**](#viz)			-- visualization methods

Classes overview
----------------

[**Cell**](#cell)			-- individual cell

[**Stimulus**](#stim)		-- a stimulus


Submodules in detail
--------------------

<h3 id="binary">rbin</h3>
<hr>
Tools for reading Igor binary files, particularly for interacting with the photodiode.

<blockquote> 
<p>`readbinhdr(fname)`
read the header from the file `fname`</p>
</blockquote>

<blockquote> 
<p>`readbin(fname, chan=0)`
read the channel `chan` from the file `fname`</p>
</blockquote>

<h3 id="spk">spktools</h3>
<hr>
Tools for manipulating spike-time arrays.

<blockquote>
<p>`binspikes(spk, binsize=0.01)`
bin spike times at the given resolution</p>
</blockquote>

<blockquote>
<p>`estimatefr(bspk, npts=9, sd=2)`
estimate firing rate by smoothing binned spikes</p>
</blockquote>

<h3 id="sta">sta</h3>
<hr>
Tools for performing spike-triggered average analyses

<blockquote>
<p>`getste(stim, vbl, spk, nframes=25)`
find the spike-triggered ensemble </p>
</blockquote>

<blockquote>
<p>`sta(stim, vbl, spk, nframes=25)`
compute the spike-triggered average </p>
</blockquote>

<blockquote>
<p>`stc(stim, vbl, spk, nframes=25)`
compute the spike-triggered covariance </p>
</blockquote>

<blockquote>
<p>`nonlin(stim, sta, nbins=30)`
compute nonlinearities</p>
</blockquote>

<h3 id="retsim">retsim</h3>
<hr>
Tools for simulating the retina.

<h3 id="viz">viz</h3>
<hr>
Visualization tools.

<blockquote>
<p>`raster(spk, trange=None)`
plot spike raster over the given time range</p>
</blockquote>

<blockquote>
<p>`psth(spk, trange=None)`
plot psth over the given time range</p>
</blockquote>

<blockquote>
<p>`playsta(sta, trange=None)`
play a spatio-temporal STA as a movie</p>
</blockquote>

Classes in detail
-----------------

<h3 id="Cell">retsim</h3>

<h3 id="Stimulus">retsim</h3>

