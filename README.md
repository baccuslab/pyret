# pyret

[![Documentation Status](https://readthedocs.org/projects/pyret/badge/?version=master)](http://pyret.readthedocs.org/en/master/?badge=master)
[![PyPi version](https://img.shields.io/pypi/v/pyret.svg)](https://pypi.python.org/pypi/pyret)

### A Python package for analyzing retinal data
Benjamin Naecker, Niru Maheswaranthan, Pablo Jadzinsky

<img src="https://cloud.githubusercontent.com/assets/904854/5329965/f91ee8e2-7d81-11e4-873f-d4253165bce9.png" height="256">

Brief description
-----------------

The pyret package provides a set of tools useful in the analysis of retina experiments
in the Baccus lab. It contains routines for manipulating spike trains, performing basic spike-triggered
analyses, and visualization tools.

Documentation
-------------
For more info and documentation, see the [pyret documentation](http://pyret.readthedocs.org/en/master/).

Demo
----
(coming soon) For a demo of how to do analysis using `pyret`, check out the html file and corresponding ipython notebook in the `demo/` folder.

Contributing
------------
Pull requests are welcome! We follow the [Numpy/Scipy documentation standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard), and [Sphinx](http://sphinx-doc.org/index.html) for generating documentation.

Testing
-------
Testing is done via [nose](https://nose.readthedocs.org/). Once installed (e.g. with `pip install nose`) then simply run `nosetests -v` at the top level directory to run the tests. Test functions are located in the `tests/` folder.
