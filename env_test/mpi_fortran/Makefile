# Compiler
FC = mpif90

# Compiler flags
FFLAGS = -fopenmp

# Source file
SRC = helloworld_mpi_openmp.f90

# Output executable
EXE = helloworld

NP = 4

# Default target
all: $(EXE)

# Rule to build the executable
$(EXE): $(SRC)
	$(FC) $(FFLAGS) $(SRC) -o $(EXE)

run: $(EXE)
	mpirun -np $(NP) ./$(EXE)

# Clean target
clean:
	rm -f $(EXE)
	rm -rf $(EXE).dSYM
