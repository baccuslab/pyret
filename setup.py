from setuptools import setup, find_packages

setup(name='pyret',
      version='0.5.3',
      description='Tools for the analysis of neural electrophysiology data',
      author='Benjamin Naecker, Niru Maheshwaranathan',
      author_email='bnaecker@stanford.edu',
      url='https://github.com/baccuslab/pyret',
      long_description='''
          The pyret package contains tools for analyzing neural
          data. In particular, it contains methods for manipulating
          spike trains (such as binning and smoothing), computing
          spike-triggered averages and ensembles, computing nonlinearities,
          as well as a suite of visualization tools.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 2.7'
      ],
      packages=find_packages(),
      install_requires=[
          'numpy>=1.11',
          'scipy>=0.18',
          'matplotlib>=1.5',
          'scikit-image>=0.12',
          'scikit-learn>=0.18'
      ],
      license='MIT',
      extras_require={
          'html': ['jupyter>=1.0']
      }
      )
