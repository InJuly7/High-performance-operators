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

__global__ void print_frag(half *A, half *B) {
    __shared__ half SMem_A[16][16];
    __shared__ half SMem_B[16][16];
    int LD_SMemA_Row = (threadIdx.x * 8) / 16;
    int LD_SMemA_Col = (threadIdx.x * 8) & 15;

    HALF8(SMem_A[LD_SMemA_Row][LD_SMemA_Col]) = HALF8(A[LD_SMemA_Row * 16 + LD_SMemA_Col]);
    HALF8(SMem_B[LD_SMemA_Row][LD_SMemA_Col]) = HALF8(B[LD_SMemA_Row * 16 + LD_SMemA_Col]);
    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> A_frag;
    // 这个位置 如果不用 wmma 的接口计算 hgemm, 设置成 col_major 还是 row_major 并不影响读取寄存器的值
    // fragment<matrix_b, 16, 16, 16, half, col_major> B_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> B_frag;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x & 31;

    uint32_t LD_SMemA_Ptr = __cvta_generic_to_shared(&SMem_A[laneId & 15][(laneId / 16) * 8]);
    LDMATRIX_X4(REG(A_frag.x[0]), REG(A_frag.x[2]), REG(A_frag.x[4]), REG(A_frag.x[6]), LD_SMemA_Ptr);
    uint32_t LD_SMemB_Ptr = __cvta_generic_to_shared(&SMem_B[laneId & 15][(laneId / 16) * 8]);
    LDMATRIX_X4(REG(B_frag.x[0]), REG(B_frag.x[2]), REG(B_frag.x[4]), REG(B_frag.x[6]), LD_SMemB_Ptr);

    for (int i = 0; i < 8; i ++) {
        cudaLog("A_frag[%d]: %f\n", i, __half2float(A_frag.x[i]));
    }

    for(int i = 0; i < 8; i++) {
        cudaLog("B_frag[%d]: %f\n", i, __half2float(B_frag.x[i]));
    }
}

int main() {
    const int M = 16;
    const int K = 16;
    const int N = 16;

    half_t *A = (half_t *)malloc(M * K * sizeof(half_t));
    half_t *B = (half_t *)malloc(K * N * sizeof(half_t));

    half *d_A = NULL, *d_B = NULL;
    cudaMalloc((void **)&d_A, M * K * sizeof(half));
    cudaMalloc((void **)&d_B, K * N * sizeof(half));


    for (int i = 0; i < M * K; i++) {
        A[i] = half_float::half_cast<half_t, int>(i);
        // std::cout << A[i] << std::endl;
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = half_float::half_cast<half_t, int>(i);
        // std::cout << B[i] << std::endl;
    }

    cudaMemcpy(d_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    dim3 grid(1);
    dim3 block(32);
    print_frag<<<grid, block>>>(d_A, d_B);
    cudaDeviceSynchronize();
    
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
}