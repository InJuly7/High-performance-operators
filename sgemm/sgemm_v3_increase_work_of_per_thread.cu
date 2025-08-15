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

/*
BLOCK: 每个 block 处理 BLOCK_SIZE*STRIDE 个元素
Threads 每个 thread 处理 STRIDE*STRIDE 个元素
*/
template <unsigned int BLOCK_SIZE, unsigned int STRIDE, unsigned int M, unsigned int N, unsigned K>
__global__ void cuda_sgemm(float *matrix_A_device, float *matrix_B_device, float *matrix_C_device) {
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    float *A_ptr_start = matrix_A_device + blockIdx.y * STEP * K;
    float *B_ptr_start = matrix_B_device + blockIdx.x * STEP;
    float *C_ptr_start = matrix_C_device + blockIdx.y * STEP * N + blockIdx.x * STEP;

    __shared__ float A_smem[STEP][STEP];
    __shared__ float B_smem[STEP][STEP];
    float temp[STRIDE][STRIDE] = {0.0f};

    for (int s = 0; s < K; s += STEP) {
        // GM --> SM
        for (int j = 0; j < STEP; j += BLOCK_SIZE) {
            for (int i = 0; i < STEP; i += BLOCK_SIZE) {
                A_smem[threadIdx.y + j][threadIdx.x + i] = A_ptr_start[(threadIdx.y + j) * K + threadIdx.x + i + s];
                B_smem[threadIdx.y + j][threadIdx.x + i] = B_ptr_start[(threadIdx.y + s + j) * N + threadIdx.x + i];
            }
        }
        __syncthreads();

        for (int j = 0; j < STRIDE; j++) {
            for (int i = 0; i < STRIDE; i++) {
                for (int k = 0; k < STEP; k++) {
                    temp[i][j] += A_smem[threadIdx.y + j * BLOCK_SIZE][k] * B_smem[k][threadIdx.x + i * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < STRIDE; j++) {
        for (int i = 0; i < STRIDE; i++) {
            C_ptr_start[(threadIdx.y + j * BLOCK_SIZE) * N + threadIdx.x + i * BLOCK_SIZE] = temp[i][j];
        }
    }
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
    const int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid(n / (BLOCK * STRIDE), m / (BLOCK * STRIDE));
    cuda_sgemm<BLOCK, STRIDE, m, n, k><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device);

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
