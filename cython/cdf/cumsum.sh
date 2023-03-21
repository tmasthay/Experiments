python setup.py build_ext --inplace > compile_message.txt 2> errors.txt
N=1000000
python -m timeit -s "from cumsum_python import cumsum" "import numpy as np" "x = np.array(np.random.random($N), dtype=np.float32)" "cumsum(x)"
python -m timeit -s "from cumsum_python_compiled import cumsum" "import numpy as np" "x = np.array(np.random.random($N), dtype=np.float32)" "cumsum(x)"
python -m timeit -s "from cumsum import cumsum" "import numpy as np" "x = np.array(np.random.random($N), dtype=np.float32)" "cumsum(x)"
python -m timeit -s "import numpy as np" "x = np.array(np.random.random($N), dtype=np.float32)" "np.cumsum(x)"
