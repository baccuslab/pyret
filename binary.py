"""
Tools for interacting with binary recording files

"""

import numpy as _np
from os import path as _path

# noinspection PyNoneFunctionAssignment
def readbin(filename, chanlist=None):
    """
    Read a binary recording file

    Parameters
    ----------
    filename : string
        Filename to read 

    chanlist : array_like
        List of channels to read. Raises IndexError if any channels are not in the file. If None (default), loads all
        channels.

    Output
    ------
    data (ndarray):
        Data from the channels requested. Shape is (nsamples, nchannels)

    """
    
    # Type of binary data, 16-bit unsigned integers
    uint = _np.dtype('>i2')

    # Check file exists
    if not _path.exists(filename):
        raise FileNotFoundError('Requested bin file {f} does not exist'.format(f=filename))

    # Read the header
    hdr = readbinhdr(filename)

    # Check the channel list given
    if chanlist is None or len(chanlist) == 0:
        chanlist = hdr['channels']
    else:
        chanlist = _np.array(chanlist)

    # Open the requested file
    with open(filename, 'rb') as fid:

        # Check all requested channels are in the file
        for chan in chanlist:
            if chan not in hdr['channels']:
                raise IndexError('Channel {c:d} is not in the file'.format(c=chan))

        # Compute number of blocks and size of each data chunk
        nblocks     = int(hdr['nsamples'] / hdr['blksize']) * uint.itemsize
        chunk_size  = hdr['nchannels'] * hdr['blksize'] * uint.itemsize
        
        # Preallocate return array
        data = _np.empty((hdr['nsamples'], len(chanlist)))

        # Loop over requested channels
        for chan in range(len(chanlist)):

            # Compute the offset into a block for this channel
            chanoffset = chanlist[chan] * hdr['blksize'] * uint.itemsize

            # Read the requested channel, a block at a time
            for block in range(nblocks):

                # Offset file position to the current block and channel
                fid.seek(hdr['hdrsize'] + block * chunk_size + chanoffset)

                # Read the data
                data[block * hdr['blksize']: (block + 1) * hdr['blksize'], chan] = _np.fromfile(fid, dtype=uint,
                                                                                                count=hdr['blksize'])

        # Scale and offset
        data *= hdr['gain']
        data += hdr['offset']

        # Return the data
        return data


# noinspection PyNoneFunctionAssignment
def readbinhdr(filename):
    """
    Read the header from a binary recording file

    Parameters
    ----------
    filename : string
        Filename to read as binary

    Returns
    -------
    hdr : dict
        Header data

    """
    
    # Define datatypes to be read in
    uint    = _np.dtype('>u4') 	# Unsigned integer, 32-bit
    short   = _np.dtype('>i2') 	# Signed 16-bit integer
    flt     = _np.dtype('>f4') 	# Float, 32-bit
    uchar   = _np.dtype('>B') 	# Unsigned char

    # Read the header
    with open(filename, 'rb') as fid:
        hdr = dict()
        hdr['hdrsize'] = _np.fromfile(fid, dtype=uint, count=1)[0] 	                # Size of header (bytes)
        hdr['type'] = _np.fromfile(fid, dtype=short, count=1)[0]	                # Not sure
        hdr['version'] = _np.fromfile(fid, dtype=short, count=1)[0] 	            # Not sure
        hdr['nsamples'] = _np.fromfile(fid, dtype=uint, count=1)[0]		            # Samples in file
        hdr['nchannels'] = _np.fromfile(fid, dtype=uint, count=1)[0] 	            # Number of channels
        hdr['channels'] = _np.fromfile(fid, dtype=short, count=hdr['nchannels'])    # Recorded channels
        hdr['fs'] = _np.fromfile(fid, dtype=flt, count=1)[0]		                # Sample rate
        hdr['blksize'] = _np.fromfile(fid, dtype=uint, count=1)[0]		            # Size of data blocks
        hdr['gain'] = _np.fromfile(fid, dtype=flt, count=1)[0]		                # Amplifier gain
        hdr['offset'] = _np.fromfile(fid, dtype=flt, count=1)[0]		            # Amplifier offset
        hdr['datesz'] = _np.fromfile(fid, dtype=uint, count=1)[0] 	                # Size of date string
        tmpdate = _np.fromfile(fid, dtype=uchar, count=hdr['datesz'])               # Date
        hdr['timesz'] = _np.fromfile(fid, dtype=uint, count=1)[0]                   # Size of time string
        tmptime = _np.fromfile(fid, dtype=uchar, count=hdr['timesz'])               # Time
        hdr['roomsz'] = _np.fromfile(fid, dtype=uint, count=1)[0]		            # Size of room string
        tmproom = _np.fromfile(fid, dtype=uchar, count=hdr['roomsz'])               # Room

        # Convert the date, time and room to strings
        hdr['date'] = ''.join([chr(i) for i in tmpdate])
        hdr['time'] = ''.join([chr(i) for i in tmptime])
        hdr['room'] = ''.join([chr(i) for i in tmproom])

        # Return the header
        return hdr