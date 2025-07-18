#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./include/util.hpp"

void cpu_sgemm(float *matrix_A_host, float *matrix_B_host, float *matrix_C_host_cpu_calc, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += matrix_A_host[m * K + k] * matrix_B_host[k * N + n];
            }
            matrix_C_host_cpu_calc[m * N + n] = temp;
        }
    }
}

template <unsigned int BLOCK_SIZE, unsigned int M, unsigned int N, unsigned K>
__global__ void cuda_sgemm(float *matrix_A_device, float *matrix_B_device, float *matrix_C_device) {
    float *A_ptr_start = matrix_A_device + blockDim.y * blockIdx.y * K;
    float *B_ptr_start = matrix_B_device + blockDim.x * blockIdx.x;
    float *C_ptr_start = matrix_C_device + blockDim.y * blockIdx.y * N + blockDim.x * blockIdx.x;

    __shared__ float A_smem[BLOCK_SIZE][K];
    __shared__ float B_smem[K][BLOCK_SIZE];

    for (int i = 0; i < K; i += BLOCK_SIZE) {
        A_smem[threadIdx.y][threadIdx.x + i] = A_ptr_start[threadIdx.y * K + threadIdx.x + i];
        B_smem[(threadIdx.y + i)][threadIdx.x] = B_ptr_start[(threadIdx.y + i) * N + threadIdx.x];
    }
    __syncthreads();

    float temp = 0.0f;
    for (int k = 0; k < K; k++) {
        temp += A_smem[threadIdx.y][k] * B_smem[k][threadIdx.x];
    }
    C_ptr_start[threadIdx.y * N + threadIdx.x] = temp;
}

int main() {
    const int m = 256, n = 256, k = 256;
    const int mem_size_A = m * k * sizeof(float);
    const int mem_size_B = k * n * sizeof(float);
    const int mem_size_C = m * n * sizeof(float);

    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);

    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    generateRandomFloatArray(matrix_A_host, m * k);
    generateRandomFloatArray(matrix_B_host, k * n);

    memset(matrix_C_host_cpu_calc, 0, mem_size_C);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);

    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);

    printFloatArray(matrix_C_host_cpu_calc, 10);

    const int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    cuda_sgemm<BLOCK, m, n, k><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device);
    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    printFloatArray(matrix_C_host_gpu_calc, 10);

    compare_matrices(m * n, matrix_C_host_cpu_calc, matrix_C_host_gpu_calc);

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu_calc);
    free(matrix_C_host_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
}
