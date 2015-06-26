#!/usr/bin/env python
'''
standard setup script. This needs at least numpy and Cython
to be installed.
'''
from setuptools import setup, Extension
from Cython.Distutils.build_ext import build_ext as _cython_build_ext
from numpy.distutils.core import \
    setup as numpy_setup, Extension as numpy_Extension
import os
import os.path
import numpy

incdirs = [numpy.get_include(), 'include', 'pyx_src', 'pyx_src/lurrn']

# depending on gcc version, use gnu++0x or c++11
CXX_STD_OPT = '-std=gnu++0x'

def replace_suffix(path, new_suffix):
    '''changes the suffix of a filename'''
    return os.path.splitext(path)[0] + new_suffix


class build_ext(_cython_build_ext):
    def cython_sources(self, sources, extension):
        return _cython_build_ext.cython_sources(self, sources, extension)

setup(name='Lurrn',
      version='0.7',
      description='Simple machine learning library',
      author='Yannick Versley',
      author_email='versley@cl.uni-heidelberg.de',
      cmdclass={'build_ext': _cython_build_ext},
      ext_modules=[Extension('lurrn.alphabet',
                             ['pyx_src/lurrn/alphabet.pyx'],
                             language='c++',
                             extra_compile_args=[CXX_STD_OPT],
                             include_dirs=incdirs),
                   Extension('lurrn.sparsmat',
                             ['pyx_src/lurrn/sparsmat.pyx',
                              'src/inverse_erf.cpp'],
                             language='c++',
                             include_dirs=incdirs),
                   Extension('lurrn.feature',
                             ['pyx_src/lurrn/feature.pyx'],
                             language='c++',
                             extra_compile_args=[CXX_STD_OPT],
                             include_dirs=incdirs),
                   Extension('lurrn.learn',
                             ['pyx_src/lurrn/learn.pyx'],
                             language='c++',
                             include_dirs=incdirs)],
      entry_points={
      },
      packages=['lurrn'],
      package_dir={'': 'py_src'}
      )
