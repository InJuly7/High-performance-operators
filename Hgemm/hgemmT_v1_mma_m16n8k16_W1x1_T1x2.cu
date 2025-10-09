#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "./include/util.hpp"
#include "../cublas/cublas.cuh"
#include "../include/cuda_log.cuh"

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
#define REG(val) (*reinterpret_cast<uint32_t *>(&(val)))

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

template <unsigned int MMA_M, unsigned int MMA_K, unsigned int MMA_N, unsigned int TM, unsigned int TN>
__global__ void hgemmT_v1_mma_m16n8k16_W1x1_T1x2(half *A, half *B, half *C, const int M, const int K, const int N) {
    const int BM = MMA_M * TM;
    const int BK = MMA_K;
    const int BN = MMA_N * TN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN * K;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ half SMem_A[BM][BK];
    __shared__ half SMem_B[BN][BK];
    uint32_t RA[4], RB[4];
    uint32_t RC[4] = {0,0};

    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    
    // (16 * 16) / 32 = 8 E/T
    int LD_GMemA_Row = (threadIdx.x * 8) / 16;
    int LD_GMemA_Col = (threadIdx.x * 8) & 15;

    // (16 * 16) / 32 = 8 E/T
    int LD_GMemB_Row = (threadIdx.x * 8) / 16;
    int LD_GMemB_Col = (threadIdx.x * 8) & 15;

    for(int k = 0; k < K; k += BK) {
        // Load GMemA/B  Store SMemA/B
        HALF8(SMem_A[LD_GMemA_Row][LD_GMemA_Col]) = HALF8(A[LD_GMemA_Row * K + LD_GMemA_Col]);
        HALF8(SMem_B[LD_GMemB_Row][LD_GMemB_Col]) = HALF8(B[LD_GMemB_Row * K + LD_GMemB_Col]);
        A += BK;
        B += BK;
        __syncthreads();
    
        // Load SMemA/B  Store RegA/B
        // x4.m8n8
        int RegA_Ptr_Row = laneId & 15;
        int RegA_Ptr_Col = (laneId / 16) * 8;
        uint32_t LD_SMemA_Ptr = __cvta_generic_to_shared(&SMem_A[RegA_Ptr_Row][RegA_Ptr_Col]);

        // x4.m8n8
        int RegB_Ptr_Row = laneId & 15;
        int RegB_Ptr_Col = (laneId / 16) * 8;
        uint32_t LD_SMemB_Ptr = __cvta_generic_to_shared(&SMem_B[RegB_Ptr_Row][RegB_Ptr_Col]);

        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], LD_SMemA_Ptr);
        LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], LD_SMemB_Ptr);
        // for(int i = 0; i < 4; i++) {
        //     float2 temp;
        //     temp = __half22float2(HALF2(RA[i]));
        //     cudaLog("RA[%d]: (%d, %d) (%f,%f)\n", i, RegA_Ptr_Row, RegA_Ptr_Col, temp.x, temp.y);
        // }

        // MMA
        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[2], RC[0], RC[1]);
        HMMA16816(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[1], RB[3], RC[2], RC[3]);
        __syncthreads();
    }

    // Store RegC  Store GMemC
    int ST_GMemC_Row = (threadIdx.x * 2) / 8;
    int ST_GMemC_Col = (threadIdx.x * 2) & 7;
    LDST32BITS(C[ST_GMemC_Row * N + ST_GMemC_Col]) = LDST32BITS(RC[0]);
    LDST32BITS(C[(ST_GMemC_Row + 8) * N + ST_GMemC_Col]) = LDST32BITS(RC[1]);
    LDST32BITS(C[ST_GMemC_Row * N + ST_GMemC_Col + 8]) = LDST32BITS(RC[2]);
    LDST32BITS(C[(ST_GMemC_Row + 8) * N + ST_GMemC_Col + 8]) = LDST32BITS(RC[3]);
}

int main() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    half_t *A = (half_t *)malloc(M * K * sizeof(half_t));
    half_t *B = (half_t *)malloc(N * K * sizeof(half_t));
    half_t *C_cublas_cal = (half_t *)malloc(M * N * sizeof(half_t));
    half_t *C_mma_cal = (half_t *)malloc(M * N * sizeof(half_t));

    generateRandomHalfArray(A, M * K, true, "A.txt");
    generateRandomHalfArray(B, N * K, true, "B.txt");

    // d_B N * K
    half *d_A, *d_B, *d_C_mma, *d_C_cublas;
    cudaMalloc((void **)&d_A, M * K * sizeof(half));
    cudaMalloc((void **)&d_B, N * K * sizeof(half));
    cudaMalloc((void **)&d_C_mma, M * N * sizeof(half));
    cudaMalloc((void **)&d_C_cublas, M * N * sizeof(half));

    cudaMemcpy(d_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(half), cudaMemcpyHostToDevice);

    // d_B N * K
    hgemmT_cublas(d_A, d_B, d_C_cublas, M, K, N);
    cudaMemcpy(C_cublas_cal, d_C_cublas, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    const int MMA_M = 16;
    const int MMA_K = 16;
    const int MMA_N = 8;
    const int TM = 1;
    const int TN = 2;
    const int BM = MMA_M * TM;
    const int BN = MMA_N * TN;
    const int BK = MMA_K;

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(32);
    hgemmT_v1_mma_m16n8k16_W1x1_T1x2<MMA_M, MMA_K, MMA_N, TM, TN><<<grid, block>>>(d_A, d_B, d_C_mma, M, K, N);
    cudaMemcpy(C_mma_cal, d_C_mma, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    printHalfArray(C_cublas_cal, 10);
    printHalfArray(C_mma_cal, 10);
    compare_matrices(M, N, C_cublas_cal, C_mma_cal);

    free(A);
    free(B);
    free(C_cublas_cal);
    free(C_mma_cal);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_mma);
    cudaFree(d_C_cublas);
    return 0;
}