#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "matrix_multiply_cpu.h"
#include "matrix_multiply_gpu.h"

int main(int argc, char const *argv[]) {

  double *a = (double*) malloc(N * M * sizeof(double));
  double *b = (double*) malloc(M * P * sizeof(double));
  double *c_cpu = (double*) malloc(N * P * sizeof(double));
  double *c_gpu = (double*) malloc(N * P * sizeof(double));

  // Initialize matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      //a[i * M + j] = sin(i + j);
      a[i * M + j] = 1.0;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      // b[i * P + j] = cos(i + j);
      b[i * P + j] = 1.0;
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

  bool correct = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      if (fabs(c_cpu[i * P + j] - c_gpu[i * P + j]) > powf(10, -12)) {
        printf("Incorrect :(\n First failure:\n");
        printf("Row: %d, Col:, %d\n", i, j);
        printf("CPU: %f\n", c_cpu[i * P + j]);
        printf("GPU: %f\n", c_gpu[i * P + j]);
        printf("Difference: %f\n", c_cpu[i * P + j] - c_gpu[i * P + j]);
        correct = 0;
        break;
      }
    }
    if (!correct) {
      break;
    }
  }

  if (correct) {
    printf("Correct!\n");
  }


  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      //printf("%f ", c_gpu[i * P + j]);
    }
    //printf("\n");
  }

  printf("TIME CPU: %lf\n", ((float)t_cpu)/CLOCKS_PER_SEC);
  printf("TIME GPU: %lf\n", ((float)t_gpu)/CLOCKS_PER_SEC);

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);

  return 0;
}
