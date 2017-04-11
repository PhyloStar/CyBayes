from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy

setup(cmdclass = {'build_ext': build_ext},
ext_modules=[ Extension("ML", ["ML.pyx"], libraries=["m"], extra_compile_args = ["-ffast-math"], include_dirs=[numpy.get_include()]), Extension("mcmc", ["mcmc.pyx"], libraries=["m"], extra_compile_args = ["-ffast-math"], include_dirs=[numpy.get_include()]), Extension("config", ["config.pyx"]), Extension("utils", ["utils.pyx"], include_dirs=[numpy.get_include()])]
)

#setup(ext_modules=cythonize(['utils.pyx', "ML.pyx", "config.pyx","mcmc.pyx"]))
