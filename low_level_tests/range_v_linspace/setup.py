# setup.py

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "float_range.pyx", 
        compiler_directives={'language_level': 3}
    )
)
