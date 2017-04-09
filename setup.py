from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension


setup(ext_modules=cythonize(['utils.pyx', "ML.pyx", "config.pyx","mcmc.pyx"]))
