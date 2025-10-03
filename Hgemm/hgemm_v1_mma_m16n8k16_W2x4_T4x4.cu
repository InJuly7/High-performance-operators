#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "./include/util.hpp"
#include "../cublas/cublas.cuh"

using namespace nvcuda;
using half_t = half_float::half;

#define WARP_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Vector Access
#define HALF2(value) (reinterpret_cast<half2 *>(&(value)))[0]
#define HALF4(value) (reinterpret_cast<float2 *>(&(value)))[0]
#define HALF8(value) (reinterpret_cast<float4 *>(&(value)))[0]
#define FLOAT2(val) (reinterpret_cast<float2 *>(&(value)))[0]
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]

#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value)))[0]
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value)))[0]
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value)))[0]

// PTX ISA
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                 \
    asm volatile(                                                           \
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, " \
        "[%4];\n"                                                           \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                            \
        : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)             \
    asm volatile(                                                               \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
        : "=r"(RD0), "=r"(RD1)                                                  \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

template <unsigned int MMA_M, unsigned int MMA_K, unsigned int MMA_N, unsigned int WARP_M, unsigned int WARP_N, unsigned int TM, unsigned int TN>
__global__ void hgemm_v1_mma_m16n8k16_W2x4_T4x4(half *A, half *B, half *C, const int M, const int K, const int N) {
    const int BM = MMA_M * WARP_M * TM;
    const int BK = MMA_K;
    const int BN = MMA_N * WARP_N * TN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ half SMem_A[BM][BK];
    __shared__ half SMem_B[BK][BN];

    // (128*16) / 256 = 8 E/T
    int LD_GMemA_Row = (threadIdx.x * 8) / 16;
    int LD_GMemA_Col = (threadIdx.x * 8) & 15;

    // (16*128) / 256 = 8 E/T
    int LD_GMemB_Row = (threadIdx.x * 8) / 128;
    int LD_GMemB_Col = (threadIdx.x * 8) & 127;

    uint32_t RA[TM][4], RB[TN][2], RC[TM][TN][2];
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            RC[tm][tn][0] = RC[tm][tn][1] = 0;
        }
    }

    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;

    int LD_SMemA_Row = ((warpId * 32) / 128) * 64;
    int LD_SMemB_Col = ((warpId * 32) & 127);
    half(*warp_tile_A)[BK] = reinterpret_cast<half(*)[BK]>(&SMem_A[LD_SMemA_Row][0]);
    half(*warp_tile_B)[BN] = reinterpret_cast<half(*)[BN]>(&SMem_B[0][LD_SMemB_Col]);
    int MMA_Row = laneId & 15;
    int MMA_Col = (laneId / 16) * 8;

    // Load GMemA/B  Store SMemA/B
    for (int k = 0; k < K; k += BK) {
        HALF8(SMem_A[LD_GMemA_Row][LD_GMemA_Col]) = HALF8(A[LD_GMemA_Row * K + LD_GMemA_Col]);
        HALF8(SMem_B[LD_GMemB_Row][LD_GMemB_Col]) = HALF8(B[LD_GMemB_Row * N + LD_GMemB_Col]);
        A += BK;
        B += BK * N;
        __syncthreads();

        for (int tm = 0; tm < TM; tm++) {
            half(*tile_A)[BK] = reinterpret_cast<half(*)[BK]>(&warp_tile_A[tm * MMA_M][0]);
            // x4.m8n8
            uint32_t LD_SMemA_Ptr = __cvta_generic_to_shared(&tile_A[MMA_Row][MMA_Col]);
            LDMATRIX_X4(RA[tm][0], RA[tm][1], RA[tm][2], RA[tm][3], LD_SMemA_Ptr);
        }

        for (int tn = 0; tn < TN; tn++) {
            half(*tile_B)[BN] = reinterpret_cast<half(*)[BN]>(&warp_tile_B[0][tn * MMA_N]);
            // x2.trans.m8n8
            uint32_t LD_SMemB_Ptr = __cvta_generic_to_shared(&tile_B[MMA_Row][0]);
            LDMATRIX_X2_T(RB[tn][0], RB[tn][1], LD_SMemB_Ptr);
        }

        for (int tm = 0; tm < TM; tm++) {
            for (int tn = 0; tn < TN; tn++) {
                HMMA16816(RC[tm][tn][0], RC[tm][tn][1], RA[tm][0], RA[tm][1], RA[tm][2], RA[tm][3], RB[tn][0], RB[tn][1], RC[tm][tn][0],
                          RC[tm][tn][1]);
            }
        }
        __syncthreads();
    }

    int ST_GMemC_Row = (laneId * 2) / 8;
    int ST_GMemC_Col = (laneId * 2) & 7;
    half *warp_tile_C = &C[LD_SMemA_Row * N + LD_SMemB_Col];
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            half *tile_C = &warp_tile_C[(tm * MMA_M) * N + tn * MMA_N];
            LDST32BITS(tile_C[ST_GMemC_Row * N + ST_GMemC_Col]) = LDST32BITS(RC[tm][tn][0]);
            LDST32BITS(tile_C[(ST_GMemC_Row + 8) * N + ST_GMemC_Col]) = LDST32BITS(RC[tm][tn][1]);
        }
    }
}

int main() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    half_t *mat_A = (half_t *)malloc(M * K * sizeof(half_t));
    half_t *mat_B = (half_t *)malloc(K * N * sizeof(half_t));
    half_t *mat_C_cublas_calc = (half_t *)malloc(M * N * sizeof(half_t));
    half_t *mat_C_mma_calc = (half_t *)malloc(M * N * sizeof(half_t));

    generateRandomHalfArray(mat_A, M * K);
    generateRandomHalfArray(mat_B, K * N);

    half *mat_A_device, *mat_B_device, *mat_C_cublas, *mat_C_mma;
    cudaMalloc((void **)&mat_A_device, M * K * sizeof(half_t));
    cudaMalloc((void **)&mat_B_device, K * N * sizeof(half_t));
    cudaMalloc((void **)&mat_C_cublas, M * N * sizeof(half_t));
    cudaMalloc((void **)&mat_C_mma, M * N * sizeof(half_t));

    cudaMemcpy(mat_A_device, mat_A, M * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B_device, mat_B, K * N * sizeof(half_t), cudaMemcpyHostToDevice);

    hgemm_cublas(mat_A_device, mat_B_device, mat_C_cublas, M, K, N);
    cudaMemcpy(mat_C_cublas_calc, mat_C_cublas, M * N * sizeof(half_t), cudaMemcpyDeviceToHost);

    const int MMA_M = 16;
    const int MMA_K = 16;
    const int MMA_N = 8;
    const int WARP_M = 2;
    const int WARP_N = 4;
    const int TILE_M = 4;
    const int TILE_N = 4;

    dim3 grid(CEIL_DIV(N, MMA_N * TILE_N * WARP_N), CEIL_DIV(M, MMA_M * TILE_M * WARP_M));
    dim3 block(256);
    for (int i = 0; i < 5; i++) {
        Perf("hgemm_v1_mma_m16n8k16_W2x4_T4x4");
        hgemm_v1_mma_m16n8k16_W2x4_T4x4<MMA_M, MMA_K, MMA_N, WARP_M, WARP_N, TILE_M, TILE_N>
            <<<grid, block>>>(mat_A_device, mat_B_device, mat_C_mma, M, K, N);
    }
    cudaMemcpy(mat_C_mma_calc, mat_C_mma, M * N * sizeof(half_t), cudaMemcpyDeviceToHost);

    printHalfArray(mat_C_cublas_calc, 10);
    printHalfArray(mat_C_mma_calc, 10);
    compare_matrices(M, N, mat_C_cublas_calc, mat_C_mma_calc);

    free(mat_A);
    free(mat_B);
    free(mat_C_cublas_calc);
    free(mat_C_mma_calc);

    cudaFree(mat_A_device);
    cudaFree(mat_B_device);
    cudaFree(mat_C_cublas);
    cudaFree(mat_C_mma);
}