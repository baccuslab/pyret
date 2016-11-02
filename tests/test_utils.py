import matplotlib
import matplotlib.pyplot as plt
from pyret.utils import plotwrapper


def test_plotwrapper():

    def dummy(**kwargs):
        pass

    f = plotwrapper(dummy)

    plt.close('all')
    fig, ax = f()
    assert type(fig) is matplotlib.figure.Figure
    assert issubclass(type(ax), matplotlib.axes.Axes)

    plt.close('all')
    fig, ax = f(fig=plt.gcf())
    assert fig is plt.gcf()
    assert issubclass(type(ax), matplotlib.axes.Axes)

    plt.close('all')
    fig, ax = f(fig=plt.gcf(), ax=plt.gca())
    assert fig is plt.gcf()
    assert ax is plt.gca()

    plt.close('all')
    _fig = plt.figure()
    _ax = _fig.add_subplot(111)
    fig, ax = f(ax=_ax)
    assert fig is _fig
    assert ax is _ax
