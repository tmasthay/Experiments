from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

ex1 = Extension("cumsum", ["cumsum.pyx"], include_dirs=[np.get_include()])
ex2 = Extension("cumsum", ["cumsum_python_compiled.py"])
ext = [ex1,ex2]

setup(ext_modules=cythonize(ext))
