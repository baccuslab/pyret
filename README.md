# pyret

### A Python package for analyzing retinal data
Benjamin Naecker, Niru Maheswaranthan, Pablo Jadzinsky

![receptive field](https://cloud.githubusercontent.com/assets/904854/5329965/f91ee8e2-7d81-11e4-873f-d4253165bce9.png)
![spikes](https://cloud.githubusercontent.com/assets/904854/5329966/f91f8090-7d81-11e4-92ea-746a659ea285.png)

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

Contributing
------------
Pull requests are welcome! We follow the [NumPy/SciPy documentation standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard), and [Sphinx](http://sphinx-doc.org/index.html) for generating documentation.

Testing
-------
Testing is done via [nose](https://nose.readthedocs.org/en/latest/). Once installed (e.g. with `pip install nose`) then simply run `nosetests -v` at the top level directory to run the tests. Test functions are located in the `tests/` folder.

Submodule overview
------------------

[**spiketools**](#spk)		-- basic manipulation of spike-times, including binning & smoothing

[**filtertools**](#sta)		-- computing components of simple linear-nonlinear models, including linear filters and nonlinearities

[**stimulustools**](#stim)	-- tools for manipulating stimuli, e.g., upsampling.

[**visualization**](#viz)	-- visualization methods

[**binary**](#binary)		-- reading binary recording files, useful for checking the photodiode

Classes overview
----------------

[**Cell**](#cell)			-- individual cell (under development)

Submodules in detail
--------------------

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

<h3 id="binary">binary</h3>
<hr>
Tools for reading Igor binary files, particularly for interacting with the photodiode.
_Note: these tools are somewhat Baccus lab specific_

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
