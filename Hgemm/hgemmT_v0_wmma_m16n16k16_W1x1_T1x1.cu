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

#define REG(val) (*reinterpret_cast<uint32_t *>(&(val)))

template<unsigned int WM, unsigned int WK, unsigned int WN>
__global__ void hgemmT_v0_wmma_m16n16k16_W1x1_T1x1(half *A, half *B, half *C, const int M, const int K, const int N) {
    
    const int BM = WM;
    const int BK = WK;
    const int BN = WN;
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN * K;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WM, WN, WK, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    for (int  k = 0; k < K; k += BK) {
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> B_frag;

        wmma::load_matrix_sync(A_frag, A, K);
        wmma::load_matrix_sync(B_frag, B, K);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        A += WK;
        B += WK;
        __syncthreads();
    }
    wmma::store_matrix_sync(C, C_frag, N, wmma::mem_row_major);
}


int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    half_t* A = (half_t*)malloc(M * K * sizeof(half_t));
    half_t* B = (half_t*)malloc(N * K * sizeof(half_t));
    half_t* C_cublas_cal = (half_t*)malloc(M * N * sizeof(half_t));
    half_t* C_wmma_cal = (half_t*)malloc(M * N * sizeof(half_t));

    generateRandomHalfArray(A, M * K);
    generateRandomHalfArray(B, N * K);

    // d_B N * K
    half *d_A, *d_B, *d_C_wmma, *d_C_cublas;
    cudaMalloc((void**)&d_A, M * K * sizeof(half));
    cudaMalloc((void**)&d_B, N * K * sizeof(half));
    cudaMalloc((void**)&d_C_wmma, M * N * sizeof(half));
    cudaMalloc((void**)&d_C_cublas, M * N * sizeof(half));

    cudaMemcpy(d_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(half), cudaMemcpyHostToDevice);

    // d_B N * K
    hgemmT_cublas(d_A, d_B, d_C_cublas, M, K, N);
    cudaMemcpy(C_cublas_cal, d_C_cublas, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    const int BM = 16;
    const int BK = 16;
    const int BN = 16;
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(32);
    hgemmT_v0_wmma_m16n16k16_W1x1_T1x1<BM, BK, BN><<<grid, block>>>(d_A, d_B, d_C_wmma, M, K, N);
    cudaMemcpy(C_wmma_cal, d_C_wmma, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    printHalfArray(C_cublas_cal, 10);
    printHalfArray(C_wmma_cal, 10);
    compare_matrices(M, N, C_cublas_cal, C_wmma_cal);

    free(A);
    free(B);
    free(C_cublas_cal);
    free(C_wmma_cal);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_wmma);
    cudaFree(d_C_cublas);
    return 0;
}