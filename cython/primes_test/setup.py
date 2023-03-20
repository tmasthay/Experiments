from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        ["primes.pyx", "primes_python_compiled.py"],
         annotate=True))
