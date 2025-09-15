#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"

#define WARP_SIZE 32
#define HALF2(val) (reinterpret_cast<half2 *>(&(val)))[0]

using half_t = half_float::half;

__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ half block_reduce_sum_f16(half val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ float warpsum[NUM_WARPS];
    val = warp_reduce_sum_f16(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    // tid == 0 返回 block_reduce_sum
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : 0.0f;
        val = warp_reduce_sum_f16(val);
    }
    return val;
}

__device__ __forceinline__ half warp_reduce_max_f16(half val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val = __hmax(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ half block_reduce_max_f16(half val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ half warpsum[NUM_WARPS];
    val = warp_reduce_max_f16(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    // tid == 0 返回 block_reduce_max
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : (half)0.0f;
        val = warp_reduce_max_f16(val);
    }
    return val;
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS ??? 没看懂,
// e^x_i/sum(e^x_0,...,e^x_n-1)
#define HALF2MAX(reg_x, reg_y) __hmax((reg_x), (reg_y))
#define HALF4MAX(reg_x, reg_y, reg_z, reg_w) __hmax(HALF2MAX(reg_x, reg_y), HALF2MAX(reg_z, reg_w))
#define HALF2_EXP(reg, global_max, local_sum) \
    (reg).x = hexp((reg).x - global_max);     \
    (reg).y = hexp((reg).y - global_max);     \
    local_sum += (reg).x;                     \
    local_sum += (reg).y;
#define HALF2_SOFTMAX(reg_B, reg_A, global_sum) \
    (reg_B).x = __hdiv((reg_A).x, global_sum);  \
    (reg_B).y = __hdiv((reg_A).y, global_sum);

template <unsigned int NUM_THREADS>
__global__ void safe_softmax_v3_f16x8_f16(half *mat_A, half *mat_B, int N) {
    half *thread_A_start = mat_A + blockIdx.x * N + 8 * threadIdx.x;
    half *thread_B_start = mat_B + blockIdx.x * N + 8 * threadIdx.x;

    __shared__ half exp_sum;
    __shared__ half global_max;

    half local_max = __float2half(-65504.0f);
    half2 reg_A_0 = HALF2(thread_A_start[0]);
    half2 reg_A_1 = HALF2(thread_A_start[2]);
    half2 reg_A_2 = HALF2(thread_A_start[4]);
    half2 reg_A_3 = HALF2(thread_A_start[6]);

    local_max = HALF2MAX(HALF4MAX(reg_A_0.x, reg_A_0.y, reg_A_1.x, reg_A_1.y), HALF4MAX(reg_A_2.x, reg_A_2.y, reg_A_3.x, reg_A_3.y));
    local_max = block_reduce_max_f16<NUM_THREADS>(local_max);
    if (threadIdx.x == 0) global_max = local_max;
    __syncthreads();

    half local_sum = __float2half(0.0f);
    HALF2_EXP(reg_A_0, global_max, local_sum);
    HALF2_EXP(reg_A_1, global_max, local_sum);
    HALF2_EXP(reg_A_2, global_max, local_sum);
    HALF2_EXP(reg_A_3, global_max, local_sum);
    local_sum = block_reduce_sum_f16<NUM_THREADS>(local_sum);
    if (threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();

    half2 reg_B_0, reg_B_1, reg_B_2, reg_B_3;
    HALF2_SOFTMAX(reg_B_0, reg_A_0, exp_sum);
    HALF2_SOFTMAX(reg_B_1, reg_A_1, exp_sum);
    HALF2_SOFTMAX(reg_B_2, reg_A_2, exp_sum);
    HALF2_SOFTMAX(reg_B_3, reg_A_3, exp_sum);
    HALF2(thread_B_start[0]) = reg_B_0;
    HALF2(thread_B_start[2]) = reg_B_1;
    HALF2(thread_B_start[4]) = reg_B_2;
    HALF2(thread_B_start[6]) = reg_B_3;
}

int main() {
    const int N1 = 4096;
    const int N2 = 1024;
    half_t *mat_A = (half_t *)malloc(N1 * N2 * sizeof(half_t));
    half_t *mat_B_cpu_calc = (half_t *)malloc(N1 * N2 * sizeof(half_t));
    generateRandomHalfArray(mat_A, N1 * N2);
    half *mat_A_device = NULL;
    cudaMalloc((void **)&mat_A_device, N1 * N2 * sizeof(half));
    cudaMemcpy(mat_A_device, mat_A, N1 * N2 * sizeof(half), cudaMemcpyHostToDevice);

    cpu_safe_softmax(mat_A, mat_B_cpu_calc, N1, N2);

    half *mat_B_device = NULL;
    half_t *mat_B_gpu_calc = (half_t *)malloc(N1 * N2 * sizeof(half_t));
    cudaMalloc((void **)&mat_B_device, N1 * N2 * sizeof(half));
    dim3 grid(N1);
    dim3 block(N2 / 8);

    for (int i = 0; i < 5; i++) {
        Perf perf("safe_softmax_v3_f16x8_f16");
        safe_softmax_v3_f16x8_f16<N2 / 8><<<grid, block>>>(mat_A_device, mat_B_device, N2);
    }

    cudaMemcpy(mat_B_gpu_calc, mat_B_device, N1 * N2 * sizeof(half), cudaMemcpyDeviceToHost);
    printHalfArray(mat_B_cpu_calc, 10);
    printHalfArray(mat_B_gpu_calc, 10);
    compare_matrices(N1, N2, mat_B_cpu_calc, mat_B_gpu_calc);

    free(mat_A);
    free(mat_B_cpu_calc);
    free(mat_B_gpu_calc);
    cudaFree(mat_A_device);
    cudaFree(mat_B_device);
}