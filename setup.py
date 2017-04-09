from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension

setup(cmdclass = {'build_ext': build_ext},
ext_modules=[ Extension("ML", ["ML.pyx"], libraries=["m"], extra_compile_args = ["-ffast-math"]), Extension("mcmc", ["mcmc.pyx"], libraries=["m"], extra_compile_args = ["-ffast-math"]), Extension("config", ["config.pyx"]), Extension("utils", ["utils.pyx"])]
)

#setup(ext_modules=cythonize(['utils.pyx', "ML.pyx", "config.pyx","mcmc.pyx"]))
