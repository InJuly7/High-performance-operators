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
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define HALF4(value) (reinterpret_cast<float2 *>(&(value))[0])
#define HALF8(val) (reinterpret_cast<float4 *>(&(val)))[0]

template <unsigned int WM, unsigned int WK, unsigned int WN, unsigned int WARP_TILE_M, unsigned int WARP_TILE_N, unsigned int TM, unsigned int TN>
__global__ void hgemm_v2_wmma_m16n16k16_W1x8_128x16(half *A, half *B, half *C, const int M, const int K, const int N) {
    const int BM = WM * WARP_TILE_M * TM;
    const int BK = WK;
    const int BN = WN * WARP_TILE_N * TN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ half SMem_A[BM][BK];
    __shared__ half SMem_B[BK][BN];

    wmma::fragment<wmma::accumulator, WM, WN, WK, half> C_frag[TM];

    for (int tm = 0; tm < TM; ++tm) {
        wmma::fill_fragment(C_frag[tm], 0.0);
    }

    // Load GMemA/B 8 E/T to SMemA/B
    // (128*16)/256 = 8
    const int LD_GMemA_Row = (threadIdx.x * 8) / 16;
    const int LD_GMemA_Col = (threadIdx.x * 8) & 15;

    const int LD_GMemB_Row = (threadIdx.x * 8) / 128;
    const int LD_GMemB_Col = (threadIdx.x * 8) & 127;

    const int warpId = threadIdx.x / 32;
    for (int k = 0; k < K; k += BK) {
        HALF8(SMem_A[LD_GMemA_Row][LD_GMemA_Col]) = HALF8(A[LD_GMemA_Row * K + LD_GMemA_Col]);
        HALF8(SMem_B[LD_GMemB_Row][LD_GMemB_Col]) = HALF8(B[LD_GMemB_Row * N + LD_GMemB_Col]);
        A += BK;
        B += BK * N;
        __syncthreads();

        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> B_frag;
        wmma::load_matrix_sync(B_frag, &SMem_B[0][warpId * WN], BN);
        // 每个 warp 计算 8个 WM*WN
        for (int tm = 0; tm < TM; tm++) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> A_frag;
            wmma::load_matrix_sync(A_frag, &SMem_A[tm * WM][0], BK);

            wmma::mma_sync(C_frag[tm], A_frag, B_frag, C_frag[tm]);
            __syncthreads();
        }
    }
    for (int tm = 0; tm < TM; ++tm) {
        wmma::store_matrix_sync(&C[tm * WM * N + warpId * WN], C_frag[tm], N, wmma::mem_row_major);
    }
}

int main() {
    const int M = 128;
    const int K = 16;
    const int N = 128;

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
    // WARP M 方向 8块, N 方向 1块
    const int TM = 8;
    const int TN = 1;
    const int WARP_TILE_M = 1;
    const int WARP_TILE_N = 8;

    dim3 grid(CEIL_DIV(N, WN * TN * WARP_TILE_N), CEIL_DIV(M, WM * TM * WARP_TILE_M));
    dim3 block(256);
    hgemm_v2_wmma_m16n16k16_W1x8_128x16<WM, WK, WN, WARP_TILE_M, WARP_TILE_N, TM, TN>
        <<<grid, block>>>(mat_A_device, mat_B_device, mat_C_wmma, M, K, N);
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