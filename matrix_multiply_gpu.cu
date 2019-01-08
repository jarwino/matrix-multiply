#include "matrix_multiply_gpu.h"

__global__ void multiply_matrices(double* a_gpu, double* b_gpu, double* c_gpu) {

  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N && j < P) {
      double sum = 0;
      for (int k = 0; k < M; k++) {
        sum += a_gpu[i * M + k] * b_gpu[k * P + j];
      }
      c_gpu[i * P + j] = sum;
  }
}

void multiply_matrices_gpu(double a[N * M], double b[M * P], double c[N * P]) {

  double* a_gpu;
  double* b_gpu;
  double* c_gpu;

  cudaMalloc(&a_gpu, N * M * sizeof(double));
  cudaMalloc(&b_gpu, M * P * sizeof(double));
  cudaMalloc(&c_gpu, N * P * sizeof(double));

  cudaMemcpy(a_gpu, a, N * M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b, M * P * sizeof(double), cudaMemcpyHostToDevice);

  int threadsPerBlock1D = 16;

  dim3 numBlocks((P + threadsPerBlock1D - 1) / threadsPerBlock1D, (N + threadsPerBlock1D - 1) / threadsPerBlock1D, 1);
  dim3 threadsPerBlock(threadsPerBlock1D, threadsPerBlock1D, 1);

  multiply_matrices<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu);

  cudaMemcpy(c, c_gpu, N * P * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(a_gpu);
  cudaFree(b_gpu);
  cudaFree(c_gpu);

}
