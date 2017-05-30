import re
import os
from setuptools import setup, find_packages


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, 'pyret/metadata.py'), 'r') as f:
    metadata = dict(re.findall("__([a-z_]+)__\s*=\s*'([^']+)'", f.read()))


setup(name='pyret',
      version=metadata['version'],
      description=metadata['description'],
      author=metadata['author'],
      author_email=metadata['author_email'],
      url=metadata['url'],
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
      license=metadata['license'],
      extras_require={
          'html': ['jupyter>=1.0']
      }
     )
