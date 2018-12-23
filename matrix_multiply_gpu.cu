#include "matrix_multiply_gpu.h"

__global__ void multiply_matrices(double* a_gpu, double* b_gpu, double* c_gpu) {
  printf("In kernel\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      double sum = 0;
      for (int k = 0; k < M; k++) {
        sum += a_gpu[i * P + k] * b_gpu[k * P + j];
      }
      c_gpu[i * P + j] = sum;
    }
  }
}

void multiply_matrices_gpu(double a[N][M], double b[M][P], double c[N][P]) {

  double* a_gpu;
  double* b_gpu;
  double* c_gpu;

  double c_flattened[N * P];

  cudaMalloc(&a_gpu, N * M * sizeof(double));
  cudaMalloc(&b_gpu, M * P * sizeof(double));
  cudaMalloc(&c_gpu, N * P * sizeof(double));

  cudaMemcpy(a_gpu, a, N * M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b, M * P * sizeof(double), cudaMemcpyHostToDevice);

  int threadsPerBlock1D = 16;

  dim3 numBlocks(N / threadsPerBlock1D, P / threadsPerBlock1D, 1);
  dim3 threadsPerBlock(threadsPerBlock1D, threadsPerBlock1D, 1);

  printf("LMAO\n");
  multiply_matrices<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu);
  printf("OMG\n");

  cudaMemcpy(c_flattened, c_gpu, N * P * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      c[i][j] = c_flattened[i * P + j];
    }
  }

  cudaFree(a_gpu);
  cudaFree(b_gpu);
  cudaFree(c_gpu);

}
