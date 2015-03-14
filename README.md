# pyret

### A Python package for analyzing retinal data
Benjamin Naecker, Niru Maheswaranthan, Pablo Jadzinsky

![receptive field](https://cloud.githubusercontent.com/assets/904854/5329965/f91ee8e2-7d81-11e4-873f-d4253165bce9.png)
![spikes](https://cloud.githubusercontent.com/assets/904854/5329966/f91f8090-7d81-11e4-92ea-746a659ea285.png)

Brief description
-----------------

The pyret package provides a set of tools useful in the analysis of retina experiments
in the Baccus lab. It contains routines for manipulating spike trains, performing basic spike-triggered
analyses, and visualization tools.

Documentation
-------------
For more info and documentation, see the [pyret website](http://baccuslab.github.io/pyret/)

Demo
----
For a demo of how to do analysis using `pyret`, check out the html file and corresponding ipython notebook in the `demo/` folder.

Contributing
------------
Pull requests are welcome! We follow the [NumPy/SciPy documentation standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard), and [Sphinx](http://sphinx-doc.org/index.html) for generating documentation.

Testing
-------
Testing is done via [nose](https://nose.readthedocs.org/en/latest/). Once installed (e.g. with `pip install nose`) then simply run `nosetests -v` at the top level directory to run the tests. Test functions are located in the `tests/` folder.
