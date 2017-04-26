#!/usr/bin/env python3
"""create_baseline_images.py
Script to create the baseline images used to test the pyret.visualizations module.
(C) 2017 The Baccus Lab
"""

import os
import functools

import utils

if __name__ == '__main__':

    # Baseline image directory
    baseline_dir = utils.get_baseline_image_dir()

    # Map from the filename to a wrapper function which will save
    # the corresponding image output from pyret.visualizations into
    # that file.
    name_to_saver = {
            'temporal-filter.png' : utils.temporal_filter_saver,
            'spatial-filter.png' : utils.spatial_filter_saver,
            'temporal-from-full-filter.png' : utils.temporal_from_spatiotemporal_filter_saver,
            'spatial-from-full-filter.png' : utils.spatial_from_spatiotemporal_filter_saver,
            'full-filter.png' : utils.spatiotemporal_filter_saver,
            'raster.png' : utils.raster_saver,
            'psth.png' : utils.psth_saver,
            'raster-and-psth.png' : utils.raster_and_psth_saver,
            'sta-movie-frame.png' : utils.sta_movie_frame_saver,
            'ellipse.png' : utils.ellipse_saver,
            'plotcells.png' : utils.plot_cells_saver,
            'rates-movie-frame.png' : utils.play_rates_saver
            }

    # Save all images
    for name, saver in name_to_saver.items():
        saver(os.path.join(baseline_dir, name))
    
