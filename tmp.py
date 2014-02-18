#!/usr/bin/env python3
#
# testing the fastste module
#
# 12 Feb 2014 bnaecker

# set path
import sys
sys.path.append('/Users/bnaecker/FileCabinet/stanford/baccuslab/projects/dataman/')
sys.path.append('/Users/bnaecker/FileCabinet/stanford/baccuslab/projects/pyret/')

import lntools, spktools, expt
import fastste

if __name__ == '__main__':
    # load an experiment and spikes
    vbl, stim, _ = expt.loadexpt('tfsens', '060214')
    spikes = spktools.loadspikes('tfsens', '060214')

    # pull out the white noise stuff
    wnstim = stim['spatialwn']
    wnvbl = vbl['spatialwnvbl']
    spk = spikes[3]

    # call fastste
    nframes = 25
    tmp = fastste.fastste(wnstim, wnvbl, spk, nframes)
    sys.exit()
