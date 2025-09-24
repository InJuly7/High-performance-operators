#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./include/util.hpp"

// block[16,16] ==> [128,128] 元素,
// BK = 8, SMem_A[128,8] SMem_B[8, 128]
// thread Load 4 元素 --> SMem

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val)))[0]
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]

template <const int BM = 128, const int BK = 8, const int BN = 128, const int TM = 8, const int TN = 8>
__global__ void sgemm_v3_t_8x8_AT_f32x4_bcf_two_line(float *mat_A, float *mat_B, float *mat_C, int M, int K, int N) {
    mat_A += blockIdx.y * BM * K;
    mat_B += blockIdx.x * BN;
    mat_C += blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ float SMem_A[BK][BM];
    __shared__ float SMem_B[BK][BN];
    // (BM * BK) / (blockDim.x) == 4;
    float ST_SMem_Reg_A[4];
    float ST_SMem_Reg_B[4];
    float LD_SMem_Reg_A[TM];
    float LD_SMem_Reg_B[TN];
    float Reg_C[TM][TN] = {0.0f};

    int LD_GMemA_row = (threadIdx.x * 4) / 8;
    int LD_GMemA_col = (threadIdx.x * 4) & 7;
    int LD_GMemB_row = (threadIdx.x * 4) / 128;
    int LD_GMemB_col = (threadIdx.x * 4) & 127;

    int ST_SMemA_row = (threadIdx.x * 4) & 7;
    int ST_SMemA_col = (threadIdx.x * 4) / 8;
    int ST_SMemB_row = (threadIdx.x * 4) / 128;
    // 组号 : (threadIdx.x * 4) / 64) * 2, 奇数多加一行 : threadIdx.x & 1
    int ST_SMemB_col = (((threadIdx.x * 4) / 64) * 2 + (threadIdx.x & 1)) * 32 + (((threadIdx.x / 2) * 4) & 31);

    int LD_SMemA_col = (threadIdx.x * 8) / 128;
    int LD_SMemB_col = (((threadIdx.x * 4) / 64) & 1) * 64 + ((threadIdx.x * 4) & 31);

    // thread block tile
    for (int k = 0; k < K; k += BK) {
        LDST128BITS(ST_SMem_Reg_A[0]) = LDST128BITS(mat_A[LD_GMemA_row * K + LD_GMemA_col]);
        LDST128BITS(ST_SMem_Reg_B[0]) = LDST128BITS(mat_B[LD_GMemB_row * N + LD_GMemB_col]);

        // LOAD 4 Float GMem_A Transpose ==> SMem_A
        // For Warp0:
        // T0: LD_row: LD_col
        SMem_A[ST_SMemA_row + 0][ST_SMemA_col] = ST_SMem_Reg_A[0];
        SMem_A[ST_SMemA_row + 1][ST_SMemA_col] = ST_SMem_Reg_A[1];
        SMem_A[ST_SMemA_row + 2][ST_SMemA_col] = ST_SMem_Reg_A[2];
        SMem_A[ST_SMemA_row + 3][ST_SMemA_col] = ST_SMem_Reg_A[3];

        // LOAD 4 Float GMem_B ==> SMem_B
        // SMem_B[LDST_row_B][LDST_col_B + 0] = ST_SMem_Reg_B[0];
        // SMem_B[LDST_row_B][LDST_col_B + 1] = ST_SMem_Reg_B[1];
        // SMem_B[LDST_row_B][LDST_col_B + 2] = ST_SMem_Reg_B[2];
        // SMem_B[LDST_row_B][LDST_col_B + 3] = ST_SMem_Reg_B[3];
        FLOAT4(SMem_B[ST_SMemB_row][ST_SMemB_col]) = FLOAT4(ST_SMem_Reg_B[0]);

        __syncthreads();
        // Translation matrix
        mat_A += BK;
        mat_B += BK * N;

        // thread tile
        // Out-product
        for (int bk = 0; bk < BK; bk++) {
            // for (int tm = 0; tm < TM; tm++) LD_SMem_Reg_A[tm] = SMem_A[k][LDST_ROW_Reg * TM + tm];
            // for (int tn = 0; tn < TN; tn++) LD_SMem_Reg_B[tn] = SMem_B[k][LDST_COL_Reg + tn];

            for (int tm = 0; tm < TM; tm += 4) FLOAT4(LD_SMem_Reg_A[tm]) = FLOAT4(SMem_A[bk][LD_SMemA_col * TM + tm]);
            FLOAT4(LD_SMem_Reg_B[0]) = FLOAT4(SMem_B[bk][LD_SMemB_col]);
            FLOAT4(LD_SMem_Reg_B[4]) = FLOAT4(SMem_B[bk][LD_SMemB_col + 32]);

            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    Reg_C[tm][tn] += LD_SMem_Reg_A[tm] * LD_SMem_Reg_B[tn];
                }
            }
        }
        __syncthreads();
    }

    // Write GMem_C
    // for (int tm = 0; tm < TM; tm++) {
    //     for (int tn = 0; tn < TN; tn++) {
    //         mat_C[(LDST_ROW_Reg * TM + tm) * N + LDST_COL_Reg + tn] = Reg_C[tm][tn];
    //     }
    // }
    int LDST_ROW_Reg = (threadIdx.x * 8) / 128;
    int LDST_COL_Reg = (threadIdx.x * 8) & 127;
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn += 4) {
            LDST128BITS(mat_C[(LDST_ROW_Reg * TM + tm) * N + LDST_COL_Reg + tn]) = LDST128BITS(Reg_C[tm][tn]);
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

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(CEIL_DIV((BN * BM), (TN * TM)));

    for (int i = 0; i < 5; i++) {
        sgemm_v3_t_8x8_AT_f32x4_bcf_two_line<BM, BK, BN, TM, TN><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_device, M, K, N);
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