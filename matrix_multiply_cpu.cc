#include "matrix_multiply_cpu.h"

void multiply_matrices_cpu(double a[N * M], double b[M * P], double c[N * P]) {

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      double sum = 0;
      for (int k = 0; k < M; k++) {
        sum += a[i * M + k] * b[k * P + j];
      }
      c[i * P + j] = sum;
    }
  }

}
