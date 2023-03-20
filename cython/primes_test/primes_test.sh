python setup.py build_ext --inplace > compile_message.txt 2> errors.txt
python -m timeit -s "from primes_python import primes" "primes(1000)"
python -m timeit -s "from primes_python_compiled import primes" "primes(1000)"
python -m timeit -s "from primes import primes" "primes(1000)"
