#!/usr/bin/env python
'''
standard setup script. This needs at least numpy and Cython
to be installed.
'''
from setuptools import setup, Extension

class lazy_cythonize(list):
    def __init__(self, callback):
        self._list = None
        self.callback = callback
    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list
    def __iter__(self):
        return iter(self.c_list())
    def __getitem__(self, ii):
        return self.c_list()[ii]
    def __len__(self):
        return len(self.c_list())

# depending on gcc version, use gnu++0x or c++11
CXX_STD_OPT = '-std=gnu++0x'

def extensions():
    try:
        from Cython.Build import cythonize
        import numpy
        incdirs = [numpy.get_include(), 'include', 'pyx_src', 'pyx_src/lurrn', '.']
    except ImportError:
        cythonize = lambda x: x
        incdirs = []
    ext_modules = [Extension('lurrn.alphabet',
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
                             include_dirs=incdirs)]
    return cythonize(ext_modules)

setup(name='Lurrn',
      version='0.7.2',
      description='Simple machine learning library',
      author='Yannick Versley',
      author_email='versley@cl.uni-heidelberg.de',
      ext_modules=lazy_cythonize(extensions),
      entry_points={
      },
      keywords=['machine learning', 'numpy'],
      packages=['lurrn'],
      package_dir={'': 'py_src'},
      install_requires=['setuptools>=17', 'cython>=0.19', 'numpy', 'simplejson'],
      )
