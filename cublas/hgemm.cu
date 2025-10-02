#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "./include/util.hpp"
#include "./cublas.cuh"

using half_t = half_float::half;

int main() {
    const int M = 128;
    const int K = 128;
    const int N = 128;

    half_t *mat_A = (half_t *)malloc(M * K * sizeof(half_t));
    half_t *mat_B = (half_t *)malloc(K * N * sizeof(half_t));
    half_t *mat_C_cpu_calc = (half_t *)malloc(M * N * sizeof(half_t));
    half_t *mat_C_gpu_calc = (half_t *)malloc(M * N * sizeof(half_t));
    
    generateRandomHalfArray(mat_A, M * K);
    generateRandomHalfArray(mat_B, K * N);

    // for(int i = 1; i <= M * K; i++) {
    //     mat_A[i] = half_float::half_cast<half_t, int>(i);
    // }

    // for(int i = 1; i <= K * N; i++) {
    //     mat_B[i] = half_float::half_cast<half_t, int>(i);
    // }

    half *mat_A_device, *mat_B_device, *mat_C_device;
    cudaMalloc((void **)&mat_A_device, M * K * sizeof(half_t));
    cudaMalloc((void **)&mat_B_device, K * N * sizeof(half_t));
    cudaMalloc((void **)&mat_C_device, M * N * sizeof(half_t));

    cudaMemcpy(mat_A_device, mat_A, M * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B_device, mat_B, K * N * sizeof(half_t), cudaMemcpyHostToDevice);

    hgemm_cublas(mat_A_device, mat_B_device, mat_C_device, M, K, N);
    cudaMemcpy(mat_C_gpu_calc, mat_C_device, M * N * sizeof(half_t), cudaMemcpyDeviceToHost);

    cpu_hgemm(mat_A, mat_B, mat_C_cpu_calc, M, K, N);
    printHalfArray(mat_C_cpu_calc, 8);
    printHalfArray(mat_C_gpu_calc, 8);
    compare_matrices(M, N, mat_C_cpu_calc, mat_C_gpu_calc);

    cudaFree(mat_A_device);
    cudaFree(mat_B_device);
    cudaFree(mat_C_device);

}