#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <string>

#include "./include/util.hpp"
#include "/home/song/program/High-performance-operators/include/half.hpp"

using half_t = half_float::half;

#define WARP_SIZE 32
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ half block_reduce_sum_f16_f16(half val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ half warpsum[NUM_WARPS];
    val = warp_reduce_sum_f16_f16(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : __float2half(0.0f);
        val = warp_reduce_sum_f16_f16(val);
    }
    return val;
}

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <unsigned int NUM_THREADS>
__global__ void rms_norm_v3_f16x2_f16(half *mat_A, half *mat_B, float g, int N, int K) {
    half *mat_A_start = mat_A + blockIdx.x * K;
    half *mat_B_start = mat_B + blockIdx.x * K;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half K_ = __int2half_rn(K);
    // 块内共享, 求出当前行 rsqrtf(sum(ai^2)/K)
    __shared__ half s_variance;

    half2 reg_A = HALF2(mat_A_start[threadIdx.x * 2]);
    half variance = reg_A.x * reg_A.x + reg_A.y * reg_A.y;
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) s_variance = hrsqrt(variance / K_ + epsilon);
    __syncthreads();
    half2 reg_B;
    reg_B.x = reg_A.x * s_variance * g_;
    reg_B.y = reg_A.y * s_variance * g_;
    HALF2(mat_B_start[threadIdx.x  *2]) = reg_B;
}

int main() {
    const int N = 4096;
    const int K = 1024;
    float g = 0.35f;

    // CPU 内存分配 - 都使用 half_t
    half_t *mat_A = (half_t *)malloc(N * K * sizeof(half_t));
    half_t *mat_B_cpu_calc = (half_t *)malloc(N * K * sizeof(half_t));
    half_t *mat_B_gpu_calc = (half_t *)malloc(N * K * sizeof(half_t));

    generateRandomHalfArray(mat_A, N * K);

    // GPU 内存分配 - 都使用 half
    half *mat_A_device = NULL;
    half *mat_B_device = NULL;
    cudaMalloc((void **)&mat_A_device, N * K * sizeof(half_t));
    cudaMalloc((void **)&mat_B_device, N * K * sizeof(half_t));
    cudaMemcpy(mat_A_device, mat_A, N * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cpu_rms_norm(mat_A, mat_B_cpu_calc, g, N, K);

    dim3 grid(N);
    dim3 block(K/2);
    for (int i = 0; i < 5; i++) {
        Perf perf("rms_norm_v3_f16x2_f16");
        rms_norm_v3_f16x2_f16<K/2><<<grid, block>>>(mat_A_device, mat_B_device, g, N, K);
    }
    cudaMemcpy(mat_B_gpu_calc, mat_B_device, N * K * sizeof(half_t), cudaMemcpyDeviceToHost);

    printHalfArray(mat_B_cpu_calc, 10);
    printHalfArray(mat_B_gpu_calc, 10);
    compare_matrices(N, K, mat_B_cpu_calc, mat_B_gpu_calc);

    free(mat_A);
    free(mat_B_cpu_calc);
    free(mat_B_gpu_calc);
    cudaFree(mat_A_device);
    cudaFree(mat_B_device);

    return 0;
}