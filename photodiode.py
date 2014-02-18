"""
Photodiode tools
author: Niru Maheswaranathan
01:07 AM Feb 6, 2014
"""

import numpy as np
from scipy.signal import correlate
from scipy.linalg import lstsq

def findDroppedFrames(pd, stim):
    """
    This function finds dropped frames in a stimulus presentation, by comparing
    the photodiode signal to the stimulus. (these are assumed to be equal

    input
    -----
    stim['time'] and stim['values'] are the Matlab (psychtoolbox) time/trace
      pd['time'] and   pd['values'] are the photodiode time/trace

    output
    ------
    A list of indices of dropped frames

    """

    # get upsample factor
    pd_dt   = np.median(np.diff(  pd['time']))
    stim_dt = np.median(np.diff(stim['time']))
    upsampleFactor = int(np.round(stim_dt / pd_dt))

    # upsample stimulus
    time_us, stim_us = upsampleStimulus(stim['time'], stim['values'], upsampleFactor)

    # Find the right scaling factor for the photodiode
    initialSlice = slice(10,100)
    weights, initialShift, rsq = findScalingFactor(pd['values'][initialSlice], stim_us[initialSlice])

    # rescale and shift the photodiode trace
    if initialShift >= 0:
        pd_time = pd['time'][initialShift:]
        pd_scaled = weights[0]*pd['values'][initialShift:] + weights[1]
    else:
        pd_time = pd['time'][:initialShift]
        pd_scaled = weights[0]*pd['values'][:initialShift] + weights[1]

    # TODO: finish loop to check for dropped frames!!
    # loop over the entire stimulus, looking for dropped frames
    #readingFrame = 100
    #for j in range(readingFrame, stim_us.size):

        # get slice of these stimuli
    return pd_time, pd_scaled, time_us, stim_us, initialShift

def findScalingFactor(x, y, numShifts = 10):
    """
    Finds a scaling factor c and time shift tau such that y(t) = m*x(t + tau) + b for all t

    """

    # how many shifts to check
    shifts = np.arange(-numShifts,numShifts)

    # initialize
    weights = list()
    residuals = list()

    # search over possible shifts
    for s in shifts:

        # compute linear regression
        w,r = linreg(x,y,s)

        # store values
        weights.append(w)
        residuals.append(r)

    # find the best rsq value
    rsq = np.array(residuals)
    idx = rsq.argmin()
    return weights[idx], shifts[idx], residuals

def linreg(x,y,shift=0):
    """
    simple least squares linear regression, with possible time shifts

    input
    -----
    x: independent variable
    y: dependent variable
    shift: possible time shift for the two arrays (in case they don't line up)

    output
    ------
    weights: regression coefficients (y = weights[0] * x + weights[1])
    rsq: sum of the squared residuals

    """

    # get shifted arrays
    if shift > 0:
        xShift = x[shift:]
        yShift = y[:xShift.size]

    elif shift < 0:
        xShift = x[:shift]
        yShift = y[-xShift.size:]

    else:
        xShift = x
        yShift = y

    weights, rsq, _, _ = lstsq(np.array([xShift, np.ones(xShift.size)]).T, yShift)
    return weights, rsq
