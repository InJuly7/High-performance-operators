#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/util.hpp"
#include "cuda_log.cuh"

// 每个Thread读2K元素, 写入1个元素
template <unsigned int BM, unsigned int BN>
__global__ void sgemm_v0_gmem_f32(float *mat_A, float *mat_B, float *mat_C, int M, int K, int N) {
    float *mat_A_start = mat_A + blockIdx.y * BM * K;
    float *mat_B_start = mat_B + blockIdx.x * BN;
    float *mat_C_start = mat_C + blockIdx.y * BM * N + blockIdx.x * BN;
    float temp = 0.0f;
    int row = threadIdx.x / 32;
    int col = threadIdx.x & 31;
    // mat_A[32,8] mat_B[8,32]
    for (int k = 0; k < K; k++) {
        temp += mat_A_start[row * K + k] * mat_B_start[k * N + col];
        cudaLog("Read Mat_A GMem[%d,%d]\n", row, k);
        cudaLog("Read Mat_B GMem[%d,%d]\n", k, col);
        // 每2次输出后同步一次，强制刷新
        if (k % 2 == 1) {
            __syncthreads();
        }
    }
    mat_C_start[row * N + col] = temp;
    cudaLog("Write Mat_C GMem[%d,%d]\n", row, col);
}

int main() {
    // 查看当前 printf 缓冲区大小
    size_t current_size;
    cudaDeviceGetLimit(&current_size, cudaLimitPrintfFifoSize);
    printf("Current printf buffer size: %zu bytes\n", current_size);
    
    // 设置更大的缓冲区
    size_t new_size = 64 * 1024 * 1024; // 64MB
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, new_size);
    cudaDeviceGetLimit(&current_size, cudaLimitPrintfFifoSize);
    printf("New printf buffer size: %zu bytes\n", current_size);

    
    const int M = 32, K = 8, N = 32;
    float *mat_A = (float *)malloc(M * K * sizeof(float));
    float *mat_B = (float *)malloc(K * N * sizeof(float));

    float *mat_C_gpu_calc = (float *)malloc(M * N * sizeof(float));
    float *mat_C_cpu_calc = (float *)malloc(M * N * sizeof(float));

    generateRandomFloatArray(mat_A, M * K);
    generateRandomFloatArray(mat_B, K * N);

    float *mat_A_device = NULL, *mat_B_device = NULL, *mat_C_device = NULL;
    cudaMalloc((void **)&mat_A_device, M * K * sizeof(float));
    cudaMalloc((void **)&mat_B_device, K * N * sizeof(float));
    cudaMalloc((void **)&mat_C_device, M * N * sizeof(float));

    cudaMemcpy(mat_A_device, mat_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B_device, mat_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cpu_sgemm(mat_A, mat_B, mat_C_cpu_calc, M, K, N);

    const int BLOCK = 32;
    dim3 block(BLOCK * BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    sgemm_v0_gmem_f32<32, 32><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_device, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(mat_C_gpu_calc, mat_C_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printFloatArray(mat_C_cpu_calc, 10);
    printFloatArray(mat_C_gpu_calc, 10);
    compare_matrices(M, N, mat_C_cpu_calc, mat_C_gpu_calc);

    free(mat_A);
    free(mat_B);
    free(mat_C_cpu_calc);
    free(mat_C_gpu_calc);

    cudaFree(mat_A_device);
    cudaFree(mat_B_device);
    cudaFree(mat_C_device);
}
