"""
Stimulus tools
author: Niru Maheswaranathan
11:49 AM Feb 10, 2014
"""

import numpy as np

def upsampleStimulus(dt_us, stim, upsampleFactor):
    """
    upsamples the stimulus by a given factor

    input
    -----
    time, stim, upsampleFactor

    output
    ------
    time_us, stim_us

    """

    # get sizes
    oldSize = stim.shape
    newSize = oldSize[:-1] + (upsampleFactor*oldSize[-1],)

    # upsample stimulus
    stim_us = (stim.reshape((-1,1))*np.ones((1,upsampleFactor))).reshape(newSize)

    # new time sampling
    time_us = np.arange(0, dt_us*stim_us.size, dt_us)

    return time_us, stim_us

def sliceStimulus(stim, history, locations=None):
    """
    slice a spatiotemporal stimulus (over time) into overlapping 'frames'

    input
    -----
    stim:       a spatiotemporal (or temporal) stimulus, with dimensions (n x n x t) or (t)
    history:    an integer number of time points to keep in each slice
    locations:  (optional) A boolean array of locations to take slices at. If not specified, all locations are used

    output
    ------
    stimSlices: A (d x t) array of stimulus slices, where d is the full (spatiotemporal) dimension of each slice, and t is the number of slices

    """

    # convert temporal stimuli to spatiotemporal
    if np.ndim(stim) == 1:
        stim = stim.reshape(1,1,-1)

    # default to using all locations
    if not locations.any():
        locations = np.ones(stim.shape[-1])

    # store stimulus slices in a big matrix
    stimSlices = np.empty((history*np.prod(stim.shape[:2]), np.sum(locations[history:])))
    stimIdx = 0

    # loop over locations
    for idx in range(history, locations.size):

        # if we need to take a slice at this location
        if locations[idx]:

            # extract the stimulus slice and store it
            stimSlices[:,stimIdx] = stim[:,:,idx-history:idx].ravel()
            stimIdx += 1

    return stimSlices
