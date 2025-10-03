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
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val)))[0]
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]

template<unsigned int WM, unsigned int WK, unsigned int WN>
__global__ void hgemm_v0_wmma_m16n16k16_W1x1_T1x1(half *A, half *B, half *C, const int M, const int K, const int N) {
    A += blockIdx.y * WM * K;
    B += blockIdx.x * WN;
    C += blockIdx.y * WM * N + blockIdx.x * WN;
    wmma::fragment<wmma::accumulator, WM, WN, WK, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    for (int k = 0; k < K; k += WK) {
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, A, K);
        wmma::load_matrix_sync(B_frag, B, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        A += WK;
        B += WK * N;
        __syncthreads();
    }
    wmma::store_matrix_sync(C, C_frag, N, wmma::mem_row_major);
}

int main() {
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
    
    dim3 grid(CEIL_DIV(N, WN), CEIL_DIV(M, WM));
    dim3 block(32);
    for(int i = 0; i < 5; i++) {
        Perf("hgemm_v0_wmma_m16n16k16_W1x1_T1x1");
        hgemm_v0_wmma_m16n16k16_W1x1_T1x1<WM, WK, WN><<<grid, block>>>(mat_A_device, mat_B_device, mat_C_wmma, M, K, N);
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