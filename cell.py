'''
cell.py

Definition of the Cell class.

(C) 2014 bnaecker, nirum
'''

import numpy as _np
import matplotlib.pyplot as _plt
from . import filtertools as _ft
from . import spiketools as _spk
from . import visualizations as _viz

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

        celltype (string):
            A string defining the cell type, e.g., 'on' or 'off'

        notes (string):
            A string with any other notes you wish

        uid (string):
            NotImplemented

        Spike data
        ----------

        These data attributes are specific to actual neural data,
        such as spike-times, linear filters, etc

        spk (ndarray):
            Spike-times

        rate (ndarray):
            Instantaneous firing rate. # NotImplemented 

        ste (ndarray):
            Spike-triggered stimulus ensemble

        sta (ndarray):
            Spike-triggered average

        staax (ndarray):
            Time axis for the STE/STA

        nonlin (ndarray):
            A cell's nonlinearity

        nonlinax (ndarray):
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
        
        Construct a Cell object, optionally with an array of spike times

        '''
        (self.ste, self.sta, self.filtax, self.nonlin) = (None for i in range(4))
        (self.celltype, self.notes, self.uid) = ('' for i in range(3))

    def addspikes(self, spk):
        '''
        
        Add spike times to the cell

        Input
        -----

        spk (ndarray):
            NumPy array of spike times

        '''
        self.spk = spk

    def getste(self, time, stim, length):
        '''
        
        Construct the spike-triggered ensemble from the given stimulus.
        If the cell contains no spike-time array, an AttributeError is 
        raised.

        Input
        -----

        time (ndarray):
            Time array, defining the time axis of the stimulus

        stim (ndarray):
            Stimulus array. first dimension should be the time
            axis of the stimulus, second should be all spatial 
            stimulus dimensions collapsed

        length (int):
            Time into the past (in frames) over which to construct the STE.

        '''

        if self.spk is None:
            # Assert that Cell contains some spikes
            print('Cell contains no spikes, please add them first')
            raise AttributeError
        else:
            # Compute the ensemble
            self.ste, self.filtax = _ft.getste(time, stim, self.spk, length)

    def getsta(self, time=None, stim=None, length=None):
        '''
        
        Compute the spike-triggered average of the cell. The function 
        has two forms.

        Form 1:
        -------

        Usage: c.getsta()

        Called with no arguments, the function returns the average of the 
        pre-constructed spike-triggered ensemble, stored in the class attribute 
        c.ste. If this attribute is None (or non-existent), an AttributeError
        is raised.

        Form 2:
        -------

        Usage: c.getsta(time, stim, length)

        In the second form, the stimulus, time array, and desired filter
        length must be passed.

        Input
        -----

        time (ndarray):
            Time array, defining the time axis of the stimulus.

        stim (ndarray):
            Stimulus array. First dimension should be time axis of the
            stimulus, and second should be all spatial dimensions collapsed

        length (int):
            Time into past (in frames) over which to construct the STA.

        '''
        # Determine which form of the function is being called
        if stim is None:
            # Try to compute the average of the STE
            if self.ste is None:
                print('Cell contains no spike-triggered ensemble. getsta must be called')
                print('in its second form, with all input arguments defined')
                raise AttributeError
            
            # STE exists, compute its average across the first dimension
            self.sta = _np.mean(self.ste, axis=0)

        else:
            # Second form of the argument, ensure all arguments given
            if not all((stim, time, length)):
                print('Using the second form of the function, the stimulus, time')
                print('array, and desired filter length must be given')
                raise ValueError
            
            # Compute the STA
            self.sta, self.filtax = _ft.getsta(time, stim, self.spk, length)

    def plot(self, time=True, space=True, ellipse=True):
        '''

        Usage: c.plot(**kwargs)
        Plot the spike-triggered average for the given cell.

        The Cell.plot() method is a very general and flexible function to plot
        the various data of the cell. Which components to plot are specified
        by keyword arguments, all of which are Boolean.

        kwargs
        ------

        time (boolean):
            Plot the temporal kernel of the receptive field.

        space (boolean):
            Plot the spatial kernel of the receptive field.

        ellipse (boolean):
            Plot a Gaussian ellipse fit to the cell's spatial receptive field.

        '''
        # Break if nothing requested
        if not any((time, space, ellipse)):
            return

        # Compute spatial and temporal kernels
        s, t = _ft.decompose(self.sta)

        # Make a figure
        fig = _plt.figure()

        # Make the right number of axes
        naxes   = sum([time, (space or ellipse)])
        axlist  = list()

        # Plot the temporal kernel
        if time:
            ax = fig.add_subplot(naxes, 1, 1)
            _viz.temporal(self.filtax, t, ax)
            axlist.append(ax)

        # Plot the spatial kernel
        if space:
            ax = fig.add_subplot(naxes, 1, 2)
            _viz.spatial(s, ax)
            axlist.append(ax)

        # Plot the ellipse
        if ellipse:
            if not self.ellipse:
                # Compute ellipse
                self.ellipse = _ft.getellipse(s)

            # Plot ellipse
            ax = fig.add_subplot(naxes, 1, 3)
            _viz.ellipse(self.ellipse, ax)
            axlist.append(ax)

    def setnotes(self, notes):
        '''
        
        Sets the notes attribute of the Cell object.
        
        '''
        self.notes = notes

    def setid(self, uid):
        '''
        
        Sets the unique id (uid) attribute of the Cell object.
        
        '''
        self.uid = uid

    def settype(self, celltype):
        '''
        
        Sets the cell-type of the Cell object
        
        '''
        self.celltype = celltype
