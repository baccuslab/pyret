from setuptools import setup, find_packages
import pyret

setup(name='pyret',
      version=pyret.__version__,
      description='Tools for the analysis of neural electrophysiology data',
      author='Benjamin Naecker, Niru Maheshwaranathan',
      author_email='bnaecker@stanford.edu',
      url='https://github.com/baccuslab/pyret',
      requires=[
              'numpy',
              'scipy', 
              'matplotlib'
              'sklearn',
              'skimage', 
          ],
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
              'numpy', 
              'scipy', 
              'matplotlib', 
              'scikit-image', 
              'scikit-learn'
          ],
      license='MIT',
      extras_require={
                'html' : ['jupyter>=1.0']
          }
      )
