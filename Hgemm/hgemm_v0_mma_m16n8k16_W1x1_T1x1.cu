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
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
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




template<unsigned int MMA_M, unsigned int MMA_K, unsigned int MMA_N>
__global__ void hgemm_v0_mma_m16n8k16_W1x1_T1x1(half *A, half *B, half *C, const int M, const int K, const int N) {
    const int BM = MMA_M;
    const int BK = MMA_K;
    const int BN = MMA_N;
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    
    __shared__ half SMem_A[BM][BK];
    __shared__ half SMem_B[BK][BN];
    // __shared__ half SMem_C[BM][BN];

    // (16*16) / 32 = 8 E/T
    int LD_GMemA_Row = (threadIdx.x * 8) / 16;
    int LD_GMemA_Col = (threadIdx.x * 8) & 15; 

    // (16*8) / 32 = 4 E/T
    int LD_GMemB_Row = (threadIdx.x * 4) / 8;
    int LD_GMemB_Col = (threadIdx.x * 4) & 7;

    uint32_t RA[4], RB[2];
    uint32_t RC[2] = {0,0};
    int laneId = threadIdx.x & (WARP_SIZE - 1);


    // Load GMemA/B  Store SMemA/B
    for(int k = 0; k < K; k += BK) {
        HALF8(SMem_A[LD_GMemA_Row][LD_GMemA_Col]) = HALF8(A[LD_GMemA_Row * K + LD_GMemA_Col]);
        HALF4(SMem_B[LD_GMemB_Row][LD_GMemB_Col]) = HALF4(B[LD_GMemB_Row * N + LD_GMemB_Col]);
        A += BK;
        B += BK * N;
        __syncthreads();
        
        // x4.m8n8
        uint32_t LD_SMemA_Ptr = __cvta_generic_to_shared(&SMem_A[laneId & 15][(laneId / 16) * 8]);      
        // x2.trans.m8n8
        uint32_t LD_SMemB_Ptr = __cvta_generic_to_shared(&SMem_B[laneId & 15][0]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], LD_SMemA_Ptr); // bfc 4 : Req 1
        LDMATRIX_X2_T(RB[0], RB[1], LD_SMemB_Ptr); // No bfc : Req 1

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        __syncthreads();
    }

    int ST_GMemC_Row = (threadIdx.x * 2) / 8;
    int ST_GMemC_Col = (threadIdx.x * 2) & 7;

    LDST32BITS(C[ST_GMemC_Row * N + ST_GMemC_Col]) = LDST32BITS(RC[0]);
    LDST32BITS(C[(ST_GMemC_Row + 8) * N + ST_GMemC_Col]) = LDST32BITS(RC[1]);
}

int main() {
    const int M = 16;
    const int K = 16;
    const int N = 8;

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

    // hgemm_cublas(mat_A_device, mat_B_device, mat_C_cublas, M, K, N);
    cudaMemcpy(mat_C_cublas_calc, mat_C_cublas, M * N * sizeof(half_t), cudaMemcpyDeviceToHost);

    const int MMA_M = 16;
    const int MMA_K = 16;
    const int MMA_N = 8;
    
    dim3 grid(CEIL_DIV(N, MMA_N), CEIL_DIV(M, MMA_M));
    dim3 block(32);
    for(int i = 0; i < 5; i++) {
        Perf("hgemm_v0_mma_m16n8k16_W1x1_T1x1");
        hgemm_v0_mma_m16n8k16_W1x1_T1x1<MMA_M, MMA_K, MMA_N><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_mma, M, K, N);
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