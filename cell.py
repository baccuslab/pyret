# cell.py
#
# Definition of Cell class in pyret package
#
# (C) 2014 bnaecker, nirum

# Imports
import scipy as sp
import matplotlib.pyplot as plt
import lntools
import spktools

class Cell:
    '''
    The Cell class bundles together useful data and functionality
    specific to individual cells recorded in an extracellular
    experiment.

    Data
    ----

    All data is stored as NumPy arrays, unless otherwise noted.

    spk:
        Spike-times
    rate:
        Instantaneous firing rate
    ste:
        Spike-triggered stimulus ensemble
    sta:
        Spike-triggered average

    Functions
    ---------

    plot:
        Plots the spike-triggered average for the cell
    getsta:
        Computes the spike-triggered average
    getste:
        Computes the spike-triggered ensemble
    psth:
        Plot a psth

    '''

    def __init__(self):
        self.spk = None
        self.ste = None
        self.sta = None

    def addspikes(self, spk):
        '''
        Usage: c.addspikes(spk)
        Add spike times to the cell

        Input:
            spk - NumPy array of spike times

        '''
        self.spk = spk

    def getste(self, stim, time, length, useC=False):
        '''
        Usage: c.getste(stim, time, length, useC=False)
        Construct the spike-triggered ensemble from the given stimulus.
        If the cell contains no spike-time array, an AttributeError is 
        raised.

        Input
        -----

        stim:
            Stimulus array. first dimension should be the time
            axis of the stimulus, second should be all spatial 
            stimulus dimensions collapsed

        time:
            Time array, defining the time axis of the stimulus

        length:
            Time into the past (in seconds) over which to construct the STE.
           
        useC:
            Boolean, describing whether or not to use the compiled
            C fastste module to construct the ensemble. Defaults to False.

        '''
        # Define the function used to compute the ensemble
        if useC:
            raise NotImplementedError
            #try:
                #stefun = fastste.fastste
            #except ImportError:
                #useC = False
                #stefun = lntools.basicste

        if self.spk is None:
            # Assert that Cell contains some spikes
            print('Cell contains no spikes, please add them first')
            raise AttributeError
        else:
            # Compute the ensemble
            self.ste = stefun(stim, time, self.spk, length)

    def getsta(self, stim=None, time=None, length=None, useC=False, saveste=True):
        '''
        Usage: c.getsta(), c.getsta(stim, time, length, useC=False, saveste=True)
        Compute the spike-triggered average of the cell. The function 
        has two forms.

        Form 1:
        ------------------

        Usage: c.getsta()

        Called with no arguments, the function returns the average of the 
        pre-constructed spike-triggered ensemble, stored in the class attribute 
        c.ste. If this attribute is None (or non-existent), an AttributeError
        is raised.

        Form 2:
        -------

        Usage: c.getsta(stim, time, length, useC=True, saveste=True)

        In the second form, the stimulus, time array, and desired filter
        length must be passed.

        Input
        -----

        stim:
            Stimulus array. First dimension should be time axis of the
            stimulus, and second should be all spatial dimensions collapsed

        time:
            Time array, defining the time axis of the stimulus.

        length:
            Time into past (in seconds) over which to construct the STA.
       
        useC:
            Boolean, describing whether to use the compiled C fastste module
            to compute the average. Defaults to False.
        
        saveste:
            Boolean. The STA is computed by constructing the spike-triggered
            ensemble, then taking the average. The parameter `saveste` defines
            whether this ste is saved as the class attribute `c.ste`.

        '''
        # Determine which form of the function is being called
        if stim is None:
            # Try to compute the average of the STE
            if self.ste is None:
                print('Cell contains no spike-triggered ensemble. getsta must be called')
                print('in its second form, with all input arguments defined')
                raise AttributeError
            
            # STE exists, compute its average across last dimension
            self.sta = sp.mean(self.ste, axis=-1)

        else:
            # Second form of the argument, ensure all arguments given
            if stim is None or time is None or length is None:
                print('Using the second form of the function, the stimulus, time')
                print('array, and desired filter length must be given')
                raise ValueError
            
            # Compute the ste
            localste = getste(stim, time, length, useC)

            # Compute its mean
            self.sta = sp.mean(localste, axis=-1)

            # Save the STE, if requested
            if saveste:
                self.ste = localste

    def plot(self):
        '''
        Usage: c.plot()
        Plot the spike-triggered average for the given cell.

        '''
        pass

    def psth(self, trange=None):
        '''
        Usage: c.psth(trange=None)
        Plot a PSTH for the cell over the given time range.

        Input
        -----

        trange:
            2-element tuple, defining time range over which to plot. If 
            None (the default), plots a PSTH over the range of the spike-time
            attribute, c.spk

        '''
        pass
