CC=clang
CFLAGS=-Wall -Wextra -Wpedantic -Xpreprocessor -fopenmp
LDFLAGS=-lomp

.PHONY: all clean

all: hello_world_openmp

hello_world_openmp: hello_world_openmp.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: hello_world_openmp
	./hello_world_openmp

clean:
	rm -f hello_world_openmp

