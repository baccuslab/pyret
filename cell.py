# cell.py
#
# Definition of Cell class in pyret package
#
# (C) 2014 bnaecker, nirum

# Imports
import numpy as np
import matplotlib.pyplot as plt
import lntools
import spktools
import viz

class Cell:
    '''
    The Cell class bundles together useful data and functionality
    specific to individual cells recorded in an extracellular
    experiment.

    Data attributes
    ---------------

        Info
        ----

        These attributes provide information about the Cell object,
        such as its cell-type, etc.

        celltype:
            A string defining the cell type, e.g., 'on' or 'off'

        notes:
            A string with any other notes you wish

        uid:
            NotImplemented

        Spike data
        ----------

        These data attributes are specific to actual neural data,
        such as spike-times, linear filters, etc

        spk:
            Spike-times

        rate:
            Instantaneous firing rate. # NotImplemented 

        ste:
            Spike-triggered stimulus ensemble

        sta:
            Spike-triggered average

        staax:
            Time axis for the STE/STA

        nonlin:
            A cell's nonlinearity

        nonlinax:
            Axis (bins) for the nonlinearity

    Function attributes
    -------------------

        Setters
        -------

        These function attributes set the "info" attributes of the
        Cell object, such as its cell-type, etc.

        settype:
            Sets the cell-type attribute

        setnotes:
            Set the notes string for the cell

        setid:
            Set the uid attribute of the cell

        Computation
        -----------

        These functions compute various attributes of the Cell object,
        such as linear filters, nonlinearities, etc.

        addspk:
            Adds an array spike times for the cell

        getsta:
            Computes the spike-triggered average

        getste:
            Computes the spike-triggered ensemble

        getstc:
            Computes the spike-triggered covariance

        getnonlin:
            Computes the nonlinearity

        Plotting
        --------

        The Cell class contains a single, flexible plotting method, simply
        called `plot`. Exactly what is plotted is controlled by keyword
        arguments. See the docstring for Cell.plot() for more information.

        plot:
            Plot various attributes of the cell, such as linear filters,
            ellipses, nonlinearity, etc.

    '''

    def __init__(self, spk=None):
        '''
        Usage: c = Cell(), c = Cell(spk)
        Construct a Cell object, optionally with an array of spike times

        '''
        (self.ste, self.sta, self.filtax, self.nonlin) = (None for i in range(4))
        (self.celltype, self.notes, self.uid) = ('' for i in range(3))

    def addspikes(self, spk):
        '''
        Usage: c.addspikes(spk)
        Add spike times to the cell

        Input
        -----
        spk:
            NumPy array of spike times

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
            self.ste, self.filtax = stefun(stim, time, self.spk, length)

    def getsta(self, stim=None, time=None, length=None, useC=False):
        '''
        Usage: c.getsta(), c.getsta(stim, time, length, useC=False)
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

        Usage: c.getsta(stim, time, length, useC=False)

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

        '''
        # Determine which form of the function is being called
        if stim is None:
            # Try to compute the average of the STE
            if self.ste is None:
                print('Cell contains no spike-triggered ensemble. getsta must be called')
                print('in its second form, with all input arguments defined')
                raise AttributeError
            
            # STE exists, compute its average across the first dimension
            self.sta = np.mean(self.ste, axis=0)

        else:
            # Second form of the argument, ensure all arguments given
            if not all((stim, time, length)):
                print('Using the second form of the function, the stimulus, time')
                print('array, and desired filter length must be given')
                raise ValueError
            
            # Compute the STA
            self.sta, self.staax = lntools.getste(stim, time, length, useC)

    def plot(self, time=True, space=True, ellipse=True):
        '''
        Usage: c.plot(**kwargs)
        Plot the spike-triggered average for the given cell.

        The Cell.plot() method is a very general and flexible function to plot
        the various data of the cell. Which components to plot are specified
        by keyword arguments, all of which are Boolean.

        kwargs
        ------

        time:
            Plot the temporal kernel of the receptive field.

        space:
            Plot the spatial kernel of the receptive field.

        ellipse:
            Plot a Gaussian ellipse fit to the cell's spatial receptive field.

        '''
        # Break if nothing requested
        if not any((time, space, ellipse)):
            return

        # Compute spatial and temporal kernels
        s, t = lntools.decompose(self.sta)

        # Make a figure
        fig = plt.figure()

        # Make the right number of axes
        naxes = sum([time, (space or ellipse)])
        axlist = fig.add_subplot(naxes, 1, 1)

        # Plot the temporal kernel
        if time:
            viz.temporal(self.staax, t, axlist[0])

        # Plot the spatial kernel
        if space:
            viz.spatial(s, axlist[-1])

        # Plot the ellipse
        if ellipse:
            if not self.ellipse:
                # Compute ellipse
                self.ellipse = filtertools.getellipse(s)
            # Plot ellipse
            viz.ellipse(self.ellipse, axlist[-1])

    def setnotes(self, notes):
        '''
        Usage: c.setnotes(notes)
        Sets the notes attribute of the Cell object.
        
        '''
        self.notes = notes

    def setid(self, uid):
        '''
        Usage: c.setid(uid)
        Sets the unique id (uid) attribute of the Cell object.
        
        '''
        self.uid = uid

    def settype(self, celltype):
        '''
        Usage: c.celltype(uid)
        Sets the cell-type of the Cell object
        
        '''
        self.celltype = celltype
