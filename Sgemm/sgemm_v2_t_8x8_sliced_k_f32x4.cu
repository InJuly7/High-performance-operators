#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <device_launch_parameters.h>
#include "./include/util.hpp"

// block[16,16] ==> [128,128] 元素,
// BK = 8, SMem_A[128,8] SMem_B[8, 128]
// thread Load 4 元素 --> SMem

#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]

template <const int BM = 128, const int BK = 8, const int BN = 128, const int TM = 8, const int TN = 8>
__global__ void sgemm_v2_t_8x8_sliced_k_f32(float *mat_A, float *mat_B, float *mat_C, int M, int K, int N) {
    float *mat_A_start = mat_A + blockIdx.y * BM * K;
    float *mat_B_start = mat_B + blockIdx.x * BN;
    float *mat_C_start = mat_C + blockIdx.y * BM * N + blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int row_A = (tid * 4) / 8;
    int col_A = (tid * 4) & 7;
    int row_B = (tid * 4) / 128;
    int col_B = (tid * 4) & 127;

    __shared__ float SMem_A[BM][BK];
    __shared__ float SMem_B[BK][BN];
    float reg_c[TM][TN] = {0.0f};
#pragma unroll
    for (int bk = 0; bk < K; bk += BK) {
        // thread[i, j] == > SMem_A : row = (tid * 4) / 8, col = (tid * 4) % 8
        // SMem_B : row = (tid * 4) / 128, col = (tid * 4) % 128
        FLOAT4(SMem_A[row_A][col_A]) = FLOAT4(mat_A_start[row_A * K + col_A + bk]);
        FLOAT4(SMem_B[row_B][col_B]) = FLOAT4(mat_B_start[(row_B + bk) * N + col_B]);
        __syncthreads();

        // out-product
        // thread[i, j] == > row = i * 8, col = j * 8
    #pragma unroll
        for (int k = 0; k < BK; k++) {
        #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    reg_c[tm][tn] += SMem_A[threadIdx.y * 8 + tm][k] * SMem_B[k][threadIdx.x * 8 + tn];
                }
            }
        }
        __syncthreads();
    }
    // thread[i,j] row : i * 8, col : j * 8
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
    #pragma unroll
        for (int tn = 0; tn < TN; tn += 4) {
            FLOAT4(mat_C_start[(threadIdx.y * 8 + tm) * N + threadIdx.x * 8 + tn]) = FLOAT4(reg_c[tm][tn]);
        }
    }
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

    const int BM = 128, BK = 8, BN = 128;
    const int TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    for (int i = 0; i < 5; i++) {
        sgemm_v2_t_8x8_sliced_k_f32<BM, BK, BN, TM, TN><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_device, M, K, N);
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