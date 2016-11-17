"""
Tests functions in the spiketools module
"""
import numpy as np
import pyret.spiketools as spk


def test_binspikes():

    # assert the proper indices are returned
    spike_times = [1.0, 2.0, 2.5, 3.0]
    dt = 0.01
    bin_edges = np.arange(0, 3, dt)
    bspk = spk.binspikes(spike_times, bin_edges)
    assert np.allclose(np.where(bspk)[0], [100, 200, 250, 299])

    # maximum absolute error is dt
    binned_times = bin_edges[np.where(bspk)]
    assert np.all(np.abs(binned_times - spike_times) <= dt)

    # test for no spikes
    assert np.allclose(spk.binspikes([], bin_edges), np.zeros_like(bin_edges))


def test_estfr():

    T = 100
    dt = 1e-2

    # test an empty array
    bspk = np.zeros(T,)
    time = np.arange(0, 1, dt)
    fr = spk.estfr(bspk, time, sigma=0.01)
    assert np.allclose(fr, bspk)

    # test a single spike
    bspk[T // 2] = 1.
    fr = spk.estfr(bspk, time, sigma=0.01)
    assert (fr.sum() * dt) == bspk.sum()


def test_spiking_events():
    np.random.seed(1234)

    # generate spike times
    spiketimes = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    N = len(spiketimes)
    T = 50
    jitter = 0.01
    spikes = []
    for trial_index in range(T):
        s = spiketimes + jitter * np.random.randn(N,)
        spikes.append(np.stack((s, trial_index * np.ones(N,))))
    spikes = np.hstack(spikes).T

    # detect events
    t, psth, bspk, events = spk.detectevents(spikes)

    # correct number of events
    assert len(events) == N

    # test SpikingEvent class
    ev = events[0]
    assert isinstance(ev, spk.SpikingEvent)

    # mean jitter should be close to the selected amount of jitter
    mean_jitter = np.mean([e.jitter() for e in events])
    assert np.allclose(mean_jitter, jitter, atol=1e-3)

    # time to first spike (TTFS) should match the only spike in each trial
    assert np.allclose(ev.spikes[:, 0], ev.ttfs())

    # one spike per trial
    mu, sigma = ev.stats()
    assert mu == 1
    assert sigma == 0

    # test sorting
    sorted_spks = ev.sort()
    sorted_spks = sorted_spks[np.argsort(sorted_spks[:, 1]), 0]
    assert np.all(np.diff(sorted_spks) > 0)


def test_peakdet():

    # create a toy signal
    u = np.linspace(-5, 5, 1001)
    x = np.exp(-u ** 2)
    dx = np.gradient(x, 1e-2)

    # one peak in x (delta=0.5)
    maxtab, mintab = spk.peakdet(x, delta=0.5)
    assert len(mintab) == 0
    assert len(maxtab) == 1
    assert np.allclose(maxtab, np.array([[500, 1]]))

    # one peak in x (delta=0.1)
    maxtab, mintab = spk.peakdet(x, delta=0.1)
    assert len(mintab) == 0
    assert len(maxtab) == 1
    assert np.allclose(maxtab, np.array([[500, 1]]))

    # no peaks in x (delta=1.0)
    maxtab, mintab = spk.peakdet(x, delta=1.)
    assert len(mintab) == 0
    assert len(maxtab) == 0

    # one peak and one valley in dx
    maxtab, mintab = spk.peakdet(dx, delta=0.2)
    assert np.allclose(maxtab, np.array([[429, 0.8576926]]))
    assert np.allclose(mintab, np.array([[571, -0.8576926]]))
