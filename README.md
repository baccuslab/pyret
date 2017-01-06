# pyret

[![Build Status](https://travis-ci.org/baccuslab/pyret.svg?branch=master)](https://travis-ci.org/baccuslab/pyret.svg?branch=master)
[![Coverage Status](https://codecov.io/gh/baccuslab/pyret/branch/master/graph/badge.svg)](https://codecov.io/gh/baccuslab/pyret)
[![Documentation Status](https://readthedocs.org/projects/pyret/badge/?version=master)](http://pyret.readthedocs.org/en/master/?badge=master)
[![PyPi version](https://img.shields.io/pypi/v/pyret.svg)](https://pypi.python.org/pypi/pyret)
[![status](http://joss.theoj.org/papers/73e486788290a6386e90a21c7e71bbe0/status.svg)](http://joss.theoj.org/papers/73e486788290a6386e90a21c7e71bbe0)

### A Python package for analyzing sensory electrophysiology data
Benjamin Naecker, Niru Maheswaranthan

<img src="https://cloud.githubusercontent.com/assets/904854/11761236/e77e2bd2-a06e-11e5-8b54-0c70f40089ab.gif" height="256">

Brief description
-----------------
The pyret package provides a set of tools for analyzing electrophysiology data, especially data recorded from sensory systems driven by an external stimulus. It was originally written and is mainly used for the analysis of multi-electrode array (MEA) recordings from the retina in the Baccus lab at Stanford University. It contains routines for manipulating spike trains, performing basic spike-triggered analyses, and visualization tools.

Documentation
-------------
For more info and documentation, see the [pyret documentation](http://pyret.readthedocs.org/en/master/).

Contributing
------------
Pull requests are welcome! We follow the [Numpy/Scipy documentation standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard), and [Sphinx](http://sphinx-doc.org/index.html) for generating documentation.

Testing
-------
Testing is done via [py.test](http://pytest.org/latest/). Once installed (e.g. with `pip install pytest`) then simply run `make test` at the top level directory to run the tests. Test functions are located in the `tests/` folder.
