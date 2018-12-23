all: main

main: main.cc matrix_multiply_cpu.o matrix_multiply_gpu.o constants.h
	nvcc -g main.cc matrix_multiply_cpu.o matrix_multiply_gpu.o -o main

matrix_multiply_cpu.o: matrix_multiply_cpu.h matrix_multiply_cpu.cc constants.h
	nvcc -c -g matrix_multiply_cpu.cc -o matrix_multiply_cpu.o

matrix_multiply_gpu.o: matrix_multiply_gpu.h matrix_multiply_gpu.cu constants.h
	nvcc -c -g matrix_multiply_gpu.cu -o matrix_multiply_gpu.o

clean:
	rm -rf *.o main
