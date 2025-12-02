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

#define WARP_SIZE 32
using half_t = half_float::half;

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
__global__ void rms_norm_v2_f16_f16(half *mat_A, half *mat_B, float g, int N, int K) {
    half *mat_A_start = mat_A + blockIdx.x * K;
    half *mat_B_start = mat_B + blockIdx.x * K;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half K_ = __int2half_rn(K);

    // 块内共享, 求出当前行 rsqrtf(sum(ai^2)/K)
    __shared__ half s_variance;
    half value = mat_A_start[threadIdx.x];
    half variance = value * value;
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) s_variance = hrsqrt(variance / K_ + epsilon);
    __syncthreads();
    mat_B_start[threadIdx.x] = (value * s_variance) * g_;
}
