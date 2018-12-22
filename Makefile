all: clean main

main: matrix_multiply_cpu.o matrix_multiply_gpu.o
	clang -g -Wall main.cc matrix_multiply_cpu.o matrix_multiply_gpu.o -o main

matrix_multiply_cpu.o: matrix_multiply_cpu.h matrix_multiply_cpu.cc
	clang -c -g -Wall matrix_multiply_cpu.cc -o matrix_multiply_cpu.o

matrix_multiply_gpu.o: matrix_multiply_gpu.h matrix_multiply_gpu.cc
	clang -c -g -Wall matrix_multiply_gpu.cc -o matrix_multiply_gpu.o

clean:
	rm -rf *.o main
