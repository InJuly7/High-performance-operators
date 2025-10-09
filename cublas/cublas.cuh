#ifndef CUBLASH_CUH 
#define CUBLASH_CUH

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <algorithm>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void hgemm_cublas(half *A, half *B, half *C, size_t M, size_t K, size_t N) {
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    half alpha = 1.0;
    half beta = 0.0;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasDestroy(handle);
}

__global__ void transpose_naive(half *d_in, half *d_out, const int N1, const int N2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * N1 + x;
    int trans_idx = x * N2 + y;
    d_out[trans_idx] = d_in[idx];
}

// A : M * K, B : N * K, BT : K * N, C : M * N
void hgemmT_cublas(half *A, half *B, half *C, size_t M, size_t K, size_t N) {
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    half alpha = 1.0;
    half beta = 0.0;
    
    half *BT; // K * N
    cudaMalloc((void **)&BT, K * N * sizeof(half));
    // B: N * K, BT: K * N
    dim3 grid(CEIL_DIV(K, 16), CEIL_DIV(N, 16));
    dim3 block(16,16);
    transpose_naive<<<grid, block>>>(B, BT, K, N);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, BT, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasDestroy(handle);
    cudaFree(BT);
}

#endif