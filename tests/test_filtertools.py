"""test_filtertools.py
Test code for pyret's filtertools module.
(C) 2016 The Baccus Lab.
"""

import numpy as np
import pytest

from pyret import filtertools as flt
from pyret.stimulustools import slicestim

import utils

def test_ste():
    """Test computing a spike-triggered ensemble."""
    np.random.seed(0)
    time = np.arange(100)
    spikes = np.array((30, 70))
    stimulus = np.random.randn(100,)
    filter_length = 5

    ste = flt.ste(time, stimulus, spikes, filter_length)
    for ix in spikes:
        assert np.allclose(stimulus[ix - filter_length : ix], next(ste))

def test_sta():
    """Test computing a spike-triggered average."""
    np.random.seed(0)
    time = np.arange(100)
    spikes = np.array((0, 30, 70))
    stimulus = np.random.randn(100,)
    filter_length = 5

    sta, tax = flt.sta(time, stimulus, spikes, filter_length)
    tmp = np.zeros(sta.shape)
    for ix in spikes[1:]: # Should ignore first spike, comes before filter_length frames
        tmp += stimulus[ix - filter_length : ix]
    tmp /= len(spikes)

    assert np.allclose(tmp, sta)
    assert np.allclose(tax, np.arange(-filter_length + 1, 1))

def test_sta_acausal():
    """Test computing a spike-triggered average with points before and
    after the time of the spike.
    """
    np.random.seed(0)
    time = np.arange(100)
    spikes = np.array((0, 30, 70))
    stimulus = np.random.randn(100,)
    nbefore, nafter = 5, 2

    sta, tax = flt.sta(time, stimulus, spikes, nbefore, nafter)
    tmp = np.zeros(sta.shape)
    for ix in spikes[1:]: # Should ignore first spike, comes before filter_length frames
        tmp += stimulus[ix - nbefore : ix + nafter]
    tmp /= len(spikes)

    assert np.allclose(tmp, sta)
    assert np.allclose(tax, np.arange(-nbefore + 1, nafter + 1))

def test_empty_sta():
    """Test that an empty with no spikes returns an array of nans"""
    np.random.seed(0)
    time = np.arange(100)
    spikes = np.array(())
    stimulus = np.random.randn(100,)
    filter_length = 5

    sta, _ = flt.sta(time, stimulus, spikes, filter_length)
    assert np.all(np.isnan(sta))

def test_stc():
    """Test computation of a spike-triggered covariance matrix."""
    np.random.seed(0)

    # random spike times an white noise stimulus, so STC should be close to identity
    npoints = 100000
    nspikes = 1000
    time = np.arange(npoints)
    spikes = np.random.randint(0, npoints, (nspikes,))
    stimulus = np.random.randn(npoints,)
    filter_length = 10

    tmp = flt.stc(time, stimulus, spikes, filter_length)
    atol = 0.1
    assert np.allclose(tmp, np.eye(filter_length), atol=atol)

def test_empty_stc():
    """Test STC with no spike returns array of nans"""
    np.random.seed(0)

    # random spike times an white noise stimulus, so STC should be close to identity
    npoints = 100
    nspikes = 0
    time = np.arange(npoints)
    spikes = np.random.randint(0, npoints, (nspikes,))
    stimulus = np.random.randn(npoints,)
    filter_length = 10

    tmp = flt.stc(time, stimulus, spikes, filter_length)
    assert np.all(np.isnan(tmp))


def test_decompose_2d():
    """Tests computing a rank-1 approximation to a 2D filter.
    Note that this tests both ``filtertools.decompose()`` and
    ``filtertools.lowranksta()``.
    """
    np.random.seed(0)
    filter_length = 50
    nx = 10
    def gaussian(x, mu, sigma):
        return np.exp(-((x - mu) / sigma)**2) / np.sqrt(sigma * 2 * np.pi)
    temporal = gaussian(np.linspace(-3, 3, filter_length), -1, 1.5)
    spatial = gaussian(np.linspace(-3, 3, nx), 0, 1.0)
    true_filter = np.outer(temporal, spatial)
    noise_std = 0.01 * (temporal.max() - temporal.min())
    true_filter += np.random.randn(*true_filter.shape) * noise_std

    s, t = flt.decompose(true_filter)
   
    # s/t are unit vectors, scale them and the inputs
    s -= s.min()
    s /= s.max()
    t -= t.min()
    t /= t.max()
    temporal -= temporal.min()
    temporal /= temporal.max()
    spatial -= spatial.min()
    spatial /= spatial.max()

    tol = 0.1
    assert np.allclose(temporal, t, atol=tol)
    assert np.allclose(spatial, s, atol=tol)


def test_decompose_3d():
    """Tests computing a rank-1 approximation to a 3D filter.
    Note that this tests both ``filtertools.decompose()`` and
    ``filtertools.lowranksta()``.
    """
    np.random.seed(0)
    filter_length = 50
    nx, ny = 10, 10
    def gaussian(x, mu, sigma):
        return np.exp(-((x - mu) / sigma)**2) / np.sqrt(sigma * 2 * np.pi)
    temporal = gaussian(np.linspace(-3, 3, filter_length), -1, 1.5)
    spatial = gaussian(np.linspace(-3, 3, nx * ny), 0, 1.0).reshape(nx, ny)
    true_filter = np.outer(temporal, spatial.ravel())
    noise_std = 0.01 * (temporal.max() - temporal.min())
    true_filter += np.random.randn(*true_filter.shape) * noise_std

    s, t = flt.decompose(true_filter)
   
    # s/t are unit vectors, scale them and the inputs
    s -= s.min()
    s /= s.max()
    t -= t.min()
    t /= t.max()
    temporal -= temporal.min()
    temporal /= temporal.max()
    spatial -= spatial.min()
    spatial /= spatial.max()

    tol = 0.1
    assert np.allclose(temporal, t, atol=tol)
    assert np.allclose(spatial.ravel(), s.ravel(), atol=tol)


def test_filterpeak():
    """Test finding the maximal point in a 3D filter"""
    arr = np.zeros((5, 2, 2))
    true_index = 7
    arr.flat[true_index] = -1
    true_indices = np.unravel_index(true_index, arr.shape)

    idx, sidx, tidx  = flt.filterpeak(arr)
    assert true_index == idx
    assert true_indices[0] == tidx
    assert np.all(true_indices[1:] == sidx)


def test_cutout():
    """Test cutting out a small tube through a 3D spatiotemporal filter"""
    np.random.seed(0)
    chunk = np.random.randn(4, 2, 2)
    arr = np.pad(chunk, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
    cutout = flt.cutout(arr, (2, 2), width=1)
    assert np.allclose(cutout, chunk)


def test_cutout_peak():
    """Test that the `filtertools.cutout()` method correctly
    uses the filter peak."""
    chunk = np.zeros((4, 2, 2))
    chunk[2, 1, 1] = 1
    arr = np.pad(chunk, ((0, 0), (1, 1), (1, 1)),
            'constant', constant_values=0)
    cutout = flt.cutout(arr, width=1)
    assert np.allclose(cutout, chunk)


def test_cutout_raises():
    """Test cutout() raises an exception when the index argument
    does not have two elements."""
    with pytest.raises(ValueError):
        flt.cutout(np.zeros((10, 10, 10)), (1,))


def test_resample():
    """Test resampling a 1 or 2D array."""
    size = 100
    arr = np.random.randn(size)
    scale = 10
    up = flt.resample(arr, scale)
    assert np.allclose(up[::scale], arr)

    orig_power = np.absolute(np.fft.fft(arr))
    up_power = np.absolute(np.fft.fft(arr))
    assert np.allclose(orig_power[: int(size / 2)], up_power[: int(size / 2)])

    arr = np.random.randn(size, size)
    up = flt.resample(arr, scale)
    assert np.allclose(up[::scale, ::scale], arr)

    orig_power = np.absolute(np.fft.fft2(arr)) * scale**2
    up_power = np.absolute(np.fft.fft2(up))
    assert np.allclose(orig_power[:int(size / 2), :int(size / 2)],
            up_power[:int(size / 2), :int(size / 2)])


def test_normalize_spatial():
    """Test normalizing a noisy filter."""
    np.random.seed(0)
    nx, ny = 10, 10
    shape = (nx, ny)
    true_filter = np.random.randn(*shape)
    noise_std = 0.01
    noisy_filter = true_filter + 1.0 * np.random.randn(*shape) * noise_std

    normalized = flt.normalize_spatial(noisy_filter)

    atol = 0.1
    assert np.allclose(normalized, true_filter, atol=atol)


def test_rfsize():
    np.random.seed(0)
    nx, ny = 10, 10
    from pyret.filtertools import _gaussian_function
    x, y = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))
    points = np.stack((x.ravel(), y.ravel()), axis=0)
    true_filter = _gaussian_function(points, 0, 0, 1, 0, 1).reshape(nx, ny)

    xsize, ysize = flt.rfsize(true_filter, 1., 1.)
    assert np.allclose(xsize, 4, 0.1) # 2SDs on either side == RF size
    assert np.allclose(ysize, 4., 0.1)


def test_linear_response_1d():
    """Test method for computing linear response from a
    filter to a one-dimensional stimulus. The linear response of the
    filter to an impulse should return the filter itself.
    """
    filt = np.array(((1, 0, 0)))
    stim = np.concatenate(((1,), np.zeros((10,))), axis=0)
    pred = flt.linear_response(filt, stim) # Impulse response is linear filter
    assert np.allclose(pred[:filt.size], filt)
    assert np.allclose(pred[filt.size:], np.zeros_like(pred[filt.size:]))


def test_linear_response_acausal():
    """Test computing a linear response from a filter to a 1D stimulus,
    including acausal portions of the stimulus. The linear response of
    the filter to an impulse should return the filter itself, plus
    zeros at any acausal time points.
    """
    nacausal_points = 1
    filt = np.concatenate((np.zeros((nacausal_points,)), (1, 0, 0)), axis=0)
    stim = np.concatenate(((1,), np.zeros((10,))), axis=0)
    pred = flt.linear_response(filt, stim, nacausal_points)
    assert np.allclose(pred[:filt.size - nacausal_points],
            filt[nacausal_points:])
    assert np.allclose(pred[filt.size:], np.zeros_like(pred[filt.size:]))


def test_linear_response_only_acausal():
    """Test that calling ``linear_response`` with only acausal
    points is invalid.
    """
    with pytest.raises(ValueError):
        flt.linear_response(np.zeros((3,)), np.zeros((10,)),
                nsamples_after=3)


def test_linear_response_nd():
    """Test method for computing linear response from a
    filter to a multi-dimensional stimulus. The linear response of
    the filter to an impulse (1 at first time point in all spatial dimensions)
    should return the filter itself, scaled by the number of spatial points.
    """
    for ndim in range(2, 4):
        filt = np.zeros((3,) + ((2,) * ndim))
        filt[0] = 1.
        stim = np.zeros((10,) + ((2,) * ndim))
        stim[0] = 1.
        pred = flt.linear_response(filt, stim)
        assert np.allclose(pred[0], filt[0].sum())
        assert np.allclose(pred[1:], np.zeros_like(pred[1:]))


def test_linear_response_raises():
    """Test raising ValueErrors with incorrect inputs"""
    with pytest.raises(ValueError):
        flt.linear_response(np.zeros((10,)), np.zeros((10,2)))
    with pytest.raises(ValueError):
        flt.linear_response(np.zeros((10, 2)), np.zeros((10, 3)))


def test_revcorr_raises():
    """Test raising ValueErrors with incorrect inputs"""
    with pytest.raises(ValueError):
        flt.revcorr(np.zeros((10, 1)), np.zeros((11,)), 2)[0]
    with pytest.raises(ValueError):
        flt.revcorr(np.zeros((10, 3)), np.zeros((10, 2)), 2)[0]


def test_revcorr_1d_ignores_beginning():
    """Verify revcorr ignores the first filter-length points of the stimulus,
    to only consider those points which the response and stimulus overlap
    completely.
    """
    filt = np.array(((1, 0, 0)))
    stim = np.concatenate(((1,), np.zeros((10,))), axis=0)
    response = np.convolve(filt, stim, 'full')[:stim.size]
    recovered, lags = flt.revcorr(stim, response, filt.size)
    assert np.allclose(recovered, 0)


def test_revcorr_1d():
    """Test computation of 1D reverse correlation.
    The reverse-correlation should recover the time-reverse of the
    linear filter, and the lags should be start at negative values
    and be strictly increasing.
    """
    filt = np.array(((1, 0, 0)))
    stim = np.zeros((10,))
    stim[5] = 1
    response = np.convolve(filt, stim, 'full')[:stim.size]
    recovered, lags = flt.revcorr(stim, response, filt.size)
    assert np.allclose(recovered, filt[::-1])
    assert lags[0] == -(filt.size - 1)
    assert (np.diff(lags) == 1).all()


def test_revcorr_acausal():
    """Test computation of a 1D linear filter by reverse correlation,
    including acausal lag values. The reverse-correlation should recover
    the time-reverse of the linear filter.
    """
    filt = np.array(((1, 0, 0)))
    stim = np.zeros((10,))
    stim[5] = 1.0
    response = np.convolve(filt, stim, 'full')[:stim.size]
    nsamples_after = 2
    recovered, lags = flt.revcorr(stim, response, filt.size, nsamples_after)
    assert np.allclose(recovered[nsamples_after:], filt[::-1])
    assert np.allclose(recovered[:filt.size], 0)
    assert lags[0] == -(filt.size - 1)
    assert lags[-1] == nsamples_after
    assert (np.diff(lags) == 1).all()


def test_revcorr_nd():
    """Test computation of 3D linear filter by reverse correlation.
    The reverse correlation should return the time-reverse of the
    linear filter, scaled by the number of spatial points.
    """
    ndim = 3
    filt = np.zeros((3,) + ((2,) * ndim))
    filt[0] = 1.
    stim = np.zeros((10,) + ((2,) * ndim))
    stim[5] = 1.
    response = flt.linear_response(filt, stim)
    recovered, lags = flt.revcorr(stim, response, filt.shape[0])
    assert np.allclose(recovered[-1], filt[0].sum())
    assert np.allclose(recovered[:-1], 0)
