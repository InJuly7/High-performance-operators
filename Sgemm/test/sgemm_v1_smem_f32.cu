#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./include/util.hpp"

// BM = BN = BK
// 每个 thread 都搬运 K/BK 元素 GMem ==> SMem
template <unsigned int M, unsigned int K, unsigned int N, unsigned int BM, unsigned BK, unsigned BN>
__global__ void sgemm_v1_smem_f32(float *mat_A, float *mat_B, float *mat_C) {
    float *mat_A_start = mat_A + blockDim.y * blockIdx.y * K;
    float *mat_B_start = mat_B + blockDim.x * blockIdx.x;
    float *mat_C_start = mat_C + blockDim.y * blockIdx.y * N + blockDim.x * blockIdx.x;

    // 计算 mat_C [BM,BN], 共享内存存储 [BM, K], [K, BN] 数据
    __shared__ float SMem_A[BM][K];
    __shared__ float SMem_B[K][BN];

    // K 整除 BM BN BK
    for (int i = 0; i < K; i += BK) {
        SMem_A[threadIdx.y][threadIdx.x + i] = mat_A_start[threadIdx.y * K + threadIdx.x + i];
        SMem_B[(threadIdx.y + i)][threadIdx.x] = mat_B_start[(threadIdx.y + i) * N + threadIdx.x];
    }
    __syncthreads();

    float temp = 0.0f;
    for (int k = 0; k < K; k++) {
        temp += SMem_A[threadIdx.y][k] * SMem_B[k][threadIdx.x];
    }
    mat_C_start[threadIdx.y * N + threadIdx.x] = temp;
}

int main() {
    const int M = 2048, K = 128, N = 2048;
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

    const int BM = 16, BK = 16, BN = 16;
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v1_smem_f32<M, K, N, BM, BK, BN><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_device);
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
