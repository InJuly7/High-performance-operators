#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <device_launch_parameters.h>
#include "../include/util.hpp"
#include "../../include/cuda_log.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val)))[0]
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]

template <const int BM = 128, const int BK = 8, const int BN = 128, const int TM = 8, const int TN = 8>
__global__ void sgemm_v3_t_8x8_AT_f32x4_bcf_1(float *mat_A, float *mat_B, float *mat_C, int M, int K, int N) {
    mat_A += blockIdx.y * BM * K;
    mat_B += blockIdx.x * BN;
    mat_C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int extraCols = 4;
    __shared__ float SMem_A[BK][BM];
    __shared__ float SMem_B[BK][BN + extraCols];
    // (BM * BK) / (blockDim.x) == 4;
    float ST_SMem_Reg_A[4];
    float ST_SMem_Reg_B[4];
    float LD_SMem_Reg_A[TM];
    float LD_SMem_Reg_B[TN];
    float Reg_C[TM][TN] = {0.0f};

    int LDST_row_A = (threadIdx.x * 4) / 8;
    int LDST_col_A = (threadIdx.x * 4) & 7;
    int LDST_row_B = (threadIdx.x * 4) / 128;
    int LDST_col_B = (threadIdx.x * 4) & 127;

    int LDST_ROW_Reg = (threadIdx.x * 8) / 128;
    int LDST_COL_Reg = (threadIdx.x * 8) % 128;

    int LDST_swizzle_row_B = (threadIdx.x * 4) & 7;
    int LDST_swizzle_col_B = ((threadIdx.x * 4) / 8) & 15;

    for (int k = 0; k < K; k += BK) {
        LDST128BITS(ST_SMem_Reg_A[0]) = LDST128BITS(mat_A[LDST_row_A * K + LDST_col_A]);
        LDST128BITS(ST_SMem_Reg_B[0]) = LDST128BITS(mat_B[LDST_row_B * N + LDST_col_B]);

        // cudaLog("Read GMem_A[%d,%d]\n", LDST_row_A, LDST_col_A);
        // cudaLog("Read GMem_B[%d,%d]\n", LDST_row_B, LDST_col_B);

        SMem_A[LDST_col_A + 0][LDST_row_A] = ST_SMem_Reg_A[0];
        SMem_A[LDST_col_A + 1][LDST_row_A] = ST_SMem_Reg_A[1];
        SMem_A[LDST_col_A + 2][LDST_row_A] = ST_SMem_Reg_A[2];
        SMem_A[LDST_col_A + 3][LDST_row_A] = ST_SMem_Reg_A[3];

        // cudaLog("Save SMem_A[%d,%d]\n", LDST_col_A + 0, LDST_row_A);
        // cudaLog("Save SMem_A[%d,%d]\n", LDST_col_A + 1, LDST_row_A);
        // cudaLog("Save SMem_A[%d,%d]\n", LDST_col_A + 2, LDST_row_A);
        // cudaLog("Save SMem_A[%d,%d]\n", LDST_col_A + 3, LDST_row_A);

        // LOAD 4 float/thread
        // 1*128 ==> row = 8 col = 16 swizzle;

        SMem_B[LDST_row_B][(LDST_swizzle_row_B + 0) * 16 + LDST_swizzle_col_B] = ST_SMem_Reg_B[0];
        SMem_B[LDST_row_B][(LDST_swizzle_row_B + 1) * 16 + LDST_swizzle_col_B] = ST_SMem_Reg_B[1];
        SMem_B[LDST_row_B][(LDST_swizzle_row_B + 2) * 16 + LDST_swizzle_col_B] = ST_SMem_Reg_B[2];
        SMem_B[LDST_row_B][(LDST_swizzle_row_B + 3) * 16 + LDST_swizzle_col_B] = ST_SMem_Reg_B[3];

        // cudaLog("Save SMem_B[%d,%d]\n", LDST_row_B, (LDST_swizzle_row_B + 0) * 16 + LDST_swizzle_col_B);
        // cudaLog("Save SMem_B[%d,%d]\n", LDST_row_B, (LDST_swizzle_row_B + 1) * 16 + LDST_swizzle_col_B);
        // cudaLog("Save SMem_B[%d,%d]\n", LDST_row_B, (LDST_swizzle_row_B + 2) * 16 + LDST_swizzle_col_B);
        // cudaLog("Save SMem_B[%d,%d]\n", LDST_row_B, (LDST_swizzle_row_B + 3) * 16 + LDST_swizzle_col_B);
        __syncthreads();

        mat_A += BK;
        mat_B += BK * N;

        LDST_swizzle_row_B = 0;
        LDST_swizzle_col_B = threadIdx.x & 15;

        for (int bk = 0; bk < BK; bk++) {
            for (int tm = 0; tm < TM; tm += 4) {
                FLOAT4(LD_SMem_Reg_A[tm]) = FLOAT4(SMem_A[bk][LDST_ROW_Reg * TM + tm]);
            }
            // cudaLog("Read SMem_A[%d,%d]\n", 0, LDST_ROW_Reg * TM + 0);
            // cudaLog("Read SMem_A[%d,%d]\n", 0, LDST_ROW_Reg * TM + 4);

            for (int tn = 0; tn < TN; tn++) {
                LD_SMem_Reg_B[tn] = SMem_B[bk][(tn * 16) + LDST_swizzle_col_B];
            }
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (0 * 16) + (threadIdx.x & 15));
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (1 * 16) + (threadIdx.x & 15));
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (2 * 16) + (threadIdx.x & 15));
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (3 * 16) + (threadIdx.x & 15));
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (4 * 16) + (threadIdx.x & 15));
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (5 * 16) + (threadIdx.x & 15));
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (6 * 16) + (threadIdx.x & 15));
            // cudaLog("Read SMem_B[%d,%d]\n", 0, (7 * 16) + (threadIdx.x & 15));

            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    Reg_C[tm][tn] += LD_SMem_Reg_A[tm] * LD_SMem_Reg_B[tn];
                }
            }
        }
        __syncthreads();
    }

    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn += 4) {
            LDST128BITS(mat_C[(LDST_ROW_Reg * TM + tm) * N + LDST_COL_Reg + tn]) = LDST128BITS(Reg_C[tm][tn]);
        }
    }
    // cudaLog("Save GMem_C[%d,%d]\n", LDST_ROW_Reg * TM + 0, LDST_COL_Reg + 0);
    // cudaLog("Save GMem_C[%d,%d]\n", LDST_ROW_Reg * TM + 7, LDST_COL_Reg + 4);
}

int main() {
    size_t new_size = 1024 * 1024 * 1024; 
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, new_size);
    const int M = 128, K = 8, N = 128;
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

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(CEIL_DIV((BN * BM), (TN * TM)));

    sgemm_v3_t_8x8_AT_f32x4_bcf_1<BM, BK, BN, TM, TN><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_device, M, K, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
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