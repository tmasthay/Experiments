CC = mpicc
CFLAGS = -Wall
TARGET = hello_world_mpi
OBJ = hello_world_mpi.o
NP = 4

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

$(OBJ): hello_world_mpi.c
	$(CC) $(CFLAGS) -c hello_world_mpi.c

run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
	rm -rf $(TARGET).dSYM

