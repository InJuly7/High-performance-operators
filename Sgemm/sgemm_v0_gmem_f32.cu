#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./include/util.hpp"

// 每个Thread读2K元素, 写入1个元素
__global__ void sgemm_v0_gmem_f32(float *mat_A, float *mat_B, float *mat_C, int M, int K, int N) {
    float *mat_A_start = mat_A + blockDim.y * blockIdx.y * K;
    float *mat_B_start = mat_B + blockDim.x * blockIdx.x;
    float *mat_C_start = mat_C + blockDim.y * blockIdx.y * N + blockDim.x * blockIdx.x;
    float temp = 0.0f;
    for (int k = 0; k < K; k++) {
        temp += mat_A_start[threadIdx.y * K + k] * mat_B_start[k * N + threadIdx.x];
    }
    mat_C_start[threadIdx.y * N + threadIdx.x] = temp;
}

int main() {
    const int M = 2048, K = 1024, N = 2048;
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
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    for(int i = 0; i < 5; i++) {
        sgemm_v0_gmem_f32<<<grid, block>>>(mat_A_device, mat_B_device, mat_C_device, M, K, N);
        cudaDeviceSynchronize();
    }
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
