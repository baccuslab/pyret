---
title: 'Pyret: A Python package for analysis of neurophysiology data'
tags:
 - neuroscience
 - sensory
 - retina
authors:
 - name: Benjamin Naecker
   orcid: 0000-0002-7525-1635
   affiliation: 1
 - name: Niru Maheswaranathan
   orcid: 0000-0002-3946-4705
   affiliation: 1
 - name: Surya Ganguli
   affiliation: 2, 3
 - name: Stephen Baccus
   affiliation: 3
affiliations:
 - name: Neurosciences Graduate Program, Stanford University
   index: 1
 - name: Department of Applied Physics, Stanford University
   index: 2
 - name: Department of Neurobiology, Stanford University
   index: 3
date: 01 Dec 2016
bibliography: paper.bib
---

# Summary

The *pyret* package contains tools for analyzing neural electrophysiology data.
It focuses on applications in sensory neuroscience, broadly construed as any experiment in which one would like to characterize neural responses to a sensory stimulus.
Pyret contains methods for manipulating spike trains (e.g. binning and smoothing), pre-processing experimental stimuli (e.g. resampling), computing spike-triggered averages and ensembles [@schwartz2006], estimating linear-nonlinear cascade models to predict neural responses to different stimuli [@chichilnisky2001], part of which follows the scikit-learn API [@pedregosa2011], as well as a suite of visualization tools for all the above.
We designed *pyret* to be simple, robust, and efficient with broad applicability across a range of sensory neuroscience analyses.

Full API documentation and a short tutorial can be found at [http://pyret.readthedocs.io/](http://pyret.readthedocs.io/)

# References
