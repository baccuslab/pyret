from setuptools import setup, find_packages
import pyret

setup(name='pyret',
      version=pyret.__version__,
      description='Tools for the analysis of neural electrophysiology data',
      author='Benjamin Naecker, Niru Maheshwaranathan',
      author_email='bnaecker@stanford.edu',
      url='https://github.com/baccuslab/pyret.git',
      requires=['scipy', 'skimage', 'numpy', 'matplotlib'],
      long_description='''
          The pyret package contains tools for analyzing neural
          data. In particular, it contains methods for manipulating
          spike trains (such as binning and smoothing), computing
          spike-triggered averages and ensembles, computing nonlinearities,
          as well as a suite of visualization tools.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      packages=find_packages(),
      license='MIT',
      )
