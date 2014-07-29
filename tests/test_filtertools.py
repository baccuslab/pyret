"""
Tests for filtertools module
author: Niru Maheswaranathan
11:03 AM Jul 29, 2014
"""
import sys
sys.path.append('../../')
import numpy as np
import pyret.filtertools as ft
import unittest

class FilterTest(unittest.TestCase):

    def setUp(self):
        # Load data for testing filter tools
        self.data = np.load('sample_data/sample_expt.npz')
        self.results = np.load('sample_data/sample_results.npz')['results']
        self.history = self.results[0]['sta'].shape[2]

    def test_getsta(self):

        # loop over cells
        for res, spk in zip(self.results, self.data['spikes']):

            # compute STA using pyret
            sta, tax = ft.getsta(self.data['time'], self.data['stimulus'], spk, self.history, norm=False)

            # verify that this matches the pre-computed STA
            assert np.allclose(sta, res['sta'])
