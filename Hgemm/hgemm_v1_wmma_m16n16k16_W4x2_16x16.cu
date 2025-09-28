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
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]

template<unsigned int WM, unsigned int WK, unsigned int WN, unsigned int TM, unsigned int TN>
__global__ void hgemm_v1_wmma_m16n16k16_W4x2_16x16(half *A, half *B, half *C, const int M, const int K, const int N) {
    const int BM = WM * TM;
    const int BK = WK;
    const int BN = WN * TN;
    
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ half SMem_A[BM][BK];
    __shared__ half SMem_B[BK][BN];
    
    const int warpId = threadIdx.x / 32;
    // const int laneId = threadIdx.x & (31);

    const int LD_GMemA_Row = (threadIdx.x * 4) / 16;
    const int LD_GMemA_Col = (threadIdx.x * 4) & 15;

    const int LD_GMemB_Row = (threadIdx.x * 2) / 32;
    const int LD_GMemB_Col = (threadIdx.x * 2) & 31;

    int LD_SMemA_Row = (warpId / TN) * WM;
    int LD_SMemB_Col = (warpId & (TN - 1)) * WN;

    wmma::fragment<wmma::accumulator, WM, WN, WK, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    for (int k = 0; k < K; k += WK) {
        HALF4(SMem_A[LD_GMemA_Row][LD_GMemA_Col]) = HALF4(A[LD_GMemA_Row * K + LD_GMemA_Col]);
        HALF2(SMem_B[LD_GMemB_Row][LD_GMemB_Col]) = HALF2(B[LD_GMemB_Row * N + LD_GMemB_Col]);
        A += WK;
        B += WK * N;
        __syncthreads();
        
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> B_frag;
        
        wmma::load_matrix_sync(A_frag, &SMem_A[LD_SMemA_Row][0], BK);
        wmma::load_matrix_sync(B_frag, &SMem_B[0][LD_SMemB_Col], BN);
        
        // C_frag += A_frag @ B_frag
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    wmma::store_matrix_sync(&C[LD_SMemA_Row * N + LD_SMemB_Col], C_frag, N, wmma::mem_row_major);
}

int main() {
    const int M = 2048;
    const int K = 2048;
    const int N = 2048;

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
    const int TM = 4;
    const int TN = 2;

    dim3 grid(CEIL_DIV(N, WN * TN), CEIL_DIV(M, WM * TM));
    dim3 block(256);
    hgemm_v1_wmma_m16n16k16_W4x2_16x16<WM, WK, WN, TM, TN><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_wmma, M, K, N);
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