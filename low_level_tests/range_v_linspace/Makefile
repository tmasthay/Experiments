# Makefile

# Change this if necessary to point to the Python interpreter you are using
PYTHON := python

# The Cython source file
CYTHON_SOURCE := float_range.pyx

# Name of the compiled shared library (will have a platform-specific extension)
COMPILED_LIB := float_range

# Default target
all: $(COMPILED_LIB)

# Build the Cython extension
$(COMPILED_LIB): $(CYTHON_SOURCE)
	$(PYTHON) setup.py build_ext --inplace

# Clean up build artifacts
clean:
	rm -rf build *.c *.so *.pyd

.PHONY: all clean
