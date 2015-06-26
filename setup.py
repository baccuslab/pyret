from setuptools import setup

setup(name = 'pyret', 
        version = '0.2',
        description = 'Tools for the analysis of neural data',
        author = 'Benjamin Naecker, Niru Maheshwaranathan',
        author_email = 'bnaecker@stanford.edu',
        url = 'https://github.com/baccuslab/pyret.git',
        requires = [i.strip() for i in open("requirements.txt").readlines()],
        long_description = '''
            The pyret package contains tools for analyzing neural
            data. In particular, it contains methods for manipulating
            spike trains (such as binning and smoothing), computing 
            spike-triggered averages and ensembles, computing nonlinearities,
            as well as a suite of visualization tools.
            ''',
        classifiers = [
            'Intended Audience :: Science/Research', 
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Scientific/Engineering :: Information Analysis'],
        packages = ['pyret'],
        package_dir = {'pyret': ''},
        py_modules = ['spiketools', 'filtertools', 'stimulustools', 'nonlinearities', 'visualizations'],
        license = 'LICENSE.md'
        )
