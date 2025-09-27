#ifndef CUBLASH_CUH 
#define CUBLASH_CUH

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <algorithm>

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

#endif