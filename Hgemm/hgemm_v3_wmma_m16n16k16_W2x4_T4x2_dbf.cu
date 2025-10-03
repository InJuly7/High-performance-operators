#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "./include/util.hpp"
#include "../cublas/cublas.cuh"

using namespace nvcuda;
using half_t = half_float::half;

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define HALF2(value) (reinterpret_cast<half2 *>(&(value)))[0]
#define HALF4(value) (reinterpret_cast<float2 *>(&(value)))[0]
#define HALF8(value) (reinterpret_cast<float4 *>(&(value)))[0]

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes
// cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

template <unsigned int WM, unsigned int WK, unsigned int WN, unsigned int WARP_M, unsigned int WARP_N, unsigned int TM, unsigned int TN>
__global__ void hgemm_v3_wmma_m16n16k16_W2x4_T4x2_dbf(half *A, half *B, half *C, const int M, const int K, const int N) {
    const int BM = WM * WARP_M * TM;
    const int BK = WK;
    const int BN = WN * WARP_N * TN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ half SMem_A[2][BM][BK];
    __shared__ half SMem_B[2][BK][BN];

    wmma::fragment<wmma::accumulator, WM, WN, WK, half> C_frag[TM][TN];

    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            wmma::fill_fragment(C_frag[tm][tn], 0.0);
        }
    }

    // Load GMemA/B 8 E/T to SMemA/B
    // (128*16)/256 = 8
    const int LD_GMemA_Row = (threadIdx.x * 8) / 16;
    const int LD_GMemA_Col = (threadIdx.x * 8) & 15;

    const int LD_GMemB_Row = (threadIdx.x * 8) / 128;
    const int LD_GMemB_Col = (threadIdx.x * 8) & 127;

    const int warpId = threadIdx.x / 32;
    const int LD_SMemA_Row = ((warpId * 32) / 128) * 64;
    const int LD_SMemB_Col = (warpId * 32) & 127;
    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> A_frag[TM];
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> B_frag[TN];

    int buffer_idx = 0;
    int buffer_next_idx = buffer_idx ^ 1;
    // HALF8(SMem_A[buffer_idx][LD_GMemA_Row][LD_GMemA_Col]) = HALF8(A[LD_GMemA_Row * K + LD_GMemA_Col]);
    // HALF8(SMem_B[buffer_idx][LD_GMemB_Row][LD_GMemB_Col]) = HALF8(B[LD_GMemB_Row * N + LD_GMemB_Col]);

    uint32_t ST_SMemA_Ptr = __cvta_generic_to_shared(&SMem_A[buffer_idx][LD_GMemA_Row][LD_GMemA_Col]);
    CP_ASYNC_CG(ST_SMemA_Ptr, &A[LD_GMemA_Row * K + LD_GMemA_Col], 16);

    uint32_t ST_SMemB_Ptr = __cvta_generic_to_shared(&SMem_B[buffer_idx][LD_GMemB_Row][LD_GMemB_Col]);
    CP_ASYNC_CG(ST_SMemB_Ptr, &B[LD_GMemB_Row * N + LD_GMemB_Col], 16);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    A += BK;
    B += BK * N;
    __syncthreads();

    for (int k = 0; k < K - BK; k += BK) {
        // HALF8(SMem_A[buffer_next_idx][LD_GMemA_Row][LD_GMemA_Col]) = HALF8(A[LD_GMemA_Row * K + LD_GMemA_Col]);
        // HALF8(SMem_B[buffer_next_idx][LD_GMemB_Row][LD_GMemB_Col]) = HALF8(B[LD_GMemB_Row * N + LD_GMemB_Col]);

        uint32_t ST_SMemA_Ptr = __cvta_generic_to_shared(&SMem_A[buffer_next_idx][LD_GMemA_Row][LD_GMemA_Col]);
        CP_ASYNC_CG(ST_SMemA_Ptr, &A[LD_GMemA_Row * K + LD_GMemA_Col], 16);

        uint32_t ST_SMemB_Ptr = __cvta_generic_to_shared(&SMem_B[buffer_next_idx][LD_GMemB_Row][LD_GMemB_Col]);
        CP_ASYNC_CG(ST_SMemB_Ptr, &B[LD_GMemB_Row * N + LD_GMemB_Col], 16);
        

        A += BK;
        B += BK * N;

        // 每个 warp 计算 TM*TN 个 WM*WN
        for (int tm = 0; tm < TM; tm++) {
            wmma::load_matrix_sync(A_frag[tm], &SMem_A[buffer_idx][LD_SMemA_Row + tm * WM][0], BK);
        }
        for (int tn = 0; tn < TN; tn++) {
            wmma::load_matrix_sync(B_frag[tn], &SMem_B[buffer_idx][0][LD_SMemB_Col + tn * WN], BN);
        }
        for (int tm = 0; tm < TM; tm++) {
            for (int tn = 0; tn < TN; tn++) {
                wmma::mma_sync(C_frag[tm][tn], A_frag[tm], B_frag[tn], C_frag[tm][tn]);
            }
        }
        buffer_idx = buffer_idx ^ 1;
        buffer_next_idx = buffer_next_idx ^ 1;
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 每个 warp 计算 TM*TN 个 WM*WN
    for (int tm = 0; tm < TM; tm++) {
        wmma::load_matrix_sync(A_frag[tm], &SMem_A[buffer_idx][LD_SMemA_Row + tm * WM][0], BK);
    }
    for (int tn = 0; tn < TN; tn++) {
        wmma::load_matrix_sync(B_frag[tn], &SMem_B[buffer_idx][0][LD_SMemB_Col + tn * WN], BN);
    }
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            wmma::mma_sync(C_frag[tm][tn], A_frag[tm], B_frag[tn], C_frag[tm][tn]);
        }
    }
    __syncthreads();

    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            wmma::store_matrix_sync(&C[(LD_SMemA_Row + tm * WM) * N + LD_SMemB_Col + tn * WN], C_frag[tm][tn], N, wmma::mem_row_major);
        }
    }
}

int main() {
    // 不知道为什么, M,N 设置成一个 block 会出现精度问题
    const int M = 1024;
    const int K = 4096;
    const int N = 1024;

    half_t *mat_A = (half_t *)malloc(M * K * sizeof(half_t));
    half_t *mat_B = (half_t *)malloc(K * N * sizeof(half_t));
    half_t *mat_C_cublas_calc = (half_t *)malloc(M * N * sizeof(half_t));
    half_t *mat_C_wmma_calc = (half_t *)malloc(M * N * sizeof(half_t));

    generateRandomHalfArray(mat_A, M * K);
    generateRandomHalfArray(mat_B, K * N);

    half *mat_A_device, *mat_B_device, *mat_C_cublas, *mat_C_wmma;
    cudaMalloc((void **)&mat_A_device, M * K * sizeof(half_t));
    cudaMalloc((void **)&mat_B_device, K * N * sizeof(half_t));
    cudaMalloc((void **)&mat_C_cublas, M * N * sizeof(half_t));
    cudaMalloc((void **)&mat_C_wmma, M * N * sizeof(half_t));

    cudaMemcpy(mat_A_device, mat_A, M * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B_device, mat_B, K * N * sizeof(half_t), cudaMemcpyHostToDevice);

    hgemm_cublas(mat_A_device, mat_B_device, mat_C_cublas, M, K, N);
    cudaMemcpy(mat_C_cublas_calc, mat_C_cublas, M * N * sizeof(half_t), cudaMemcpyDeviceToHost);

    const int WM = 16;
    const int WK = 16;
    const int WN = 16;
    // WARP 2x4
    const int WARP_M = 2;
    const int WARP_N = 4;
    // Tile M 方向 4块, N 方向 2块
    const int TM = 4;
    const int TN = 2;

    dim3 grid(CEIL_DIV(N, WN * TN * WARP_N), CEIL_DIV(M, WM * TM * WARP_M));
    dim3 block(256);

    for (int i = 0; i < 5; i++) {
        Perf("hgemm_v3_wmma_m16n16k16_W2x4_T4x2_dbf");
        hgemm_v3_wmma_m16n16k16_W2x4_T4x2_dbf<WM, WK, WN, WARP_M, WARP_N, TM, TN>
            <<<grid, block>>>(mat_A_device, mat_B_device, mat_C_wmma, M, K, N);
    }
    cudaMemcpy(mat_C_wmma_calc, mat_C_wmma, M * N * sizeof(half_t), cudaMemcpyDeviceToHost);

    printHalfArray(mat_C_cublas_calc, 10);
    printHalfArray(mat_C_wmma_calc, 10);
    compare_matrices(M, N, mat_C_cublas_calc, mat_C_wmma_calc);

    free(mat_A);
    free(mat_B);
    free(mat_C_cublas_calc);
    free(mat_C_wmma_calc);

    cudaFree(mat_A_device);
    cudaFree(mat_B_device);
    cudaFree(mat_C_cublas);
    cudaFree(mat_C_wmma);
}