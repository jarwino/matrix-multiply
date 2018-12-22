#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "matrix_multiply_cpu.h"
#include "matrix_multiply_gpu.h"

int main(int argc, char const *argv[]) {

  double a[N][M];
  double b[M][P];
  double c_cpu[N][P];
  double c_gpu[N][P];

  // Initialize matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      a[i][j] = 7.0;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      b[i][j] = 9.0;
    }
  }

  // Calculate CPU performance
  clock_t t_cpu = clock();
  for (size_t i = 0; i < NUM_TRIALS; i++) {
    multiply_matrices_cpu(a, b, c_cpu);
  }
  t_cpu = clock() - t_cpu;
  t_cpu = (float)t_cpu / NUM_TRIALS;

  // Calculate GPU performance
  clock_t t_gpu = clock();
  for (size_t i = 0; i < NUM_TRIALS; i++) {
    multiply_matrices_gpu(a, b, c_gpu);
  }
  t_gpu = clock() - t_gpu;
  t_gpu = (float)t_gpu / NUM_TRIALS;

/*
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      printf("%f ", c[i][j]);
    }
    printf("\n");
  }
*/

  printf("TIME CPU: %lf\n", ((float)t_cpu)/CLOCKS_PER_SEC);
  printf("TIME GPU: %lf\n", ((float)t_gpu)/CLOCKS_PER_SEC);

  return 0;
}
