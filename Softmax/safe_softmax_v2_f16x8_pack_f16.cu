#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>

#include <random>
#include <string>
#include "./include/util.hpp"

#define WARP_SIZE 32

#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val)))[0]

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
template <unsigned int NUM_THREADS>
__global__ void safe_softmax_v2_f16x8_pack_f16(half *mat_A, half *mat_B, int N) {
    half *thread_A_start = mat_A + blockIdx.x * N + 8 * threadIdx.x;
    half *thread_B_start = mat_B + blockIdx.x * N + 8 * threadIdx.x;
    half pack_A[8], pack_B[8];
    LDST128BITS(pack_A[0]) = LDST128BITS(thread_A_start[0]);
    __shared__ half exp_sum;
    __shared__ half global_max;

    half local_max = __float2half(-65504.0f);
    for(int i = 0; i < 8; i++) {
        local_max = __hmax(local_max, pack_A[i]);
    }
    local_max = block_reduce_max_f16<NUM_THREADS>(local_max);
    if(threadIdx.x == 0) global_max = local_max;
    __syncthreads();

    half local_sum = __float2half(0.0f);
    for(int i = 0; i < 8; i++) {
        pack_A[i] = hexp(pack_A[i] - global_max);
        local_sum += pack_A[i];
    }
    local_sum = block_reduce_sum_f16<NUM_THREADS>(local_sum);
    if(threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();
    
    for(int i = 0; i < 8; i++) {
        pack_B[i] = pack_A[i]/exp_sum;
    }
    LDST128BITS(thread_B_start[0]) = LDST128BITS(pack_B[0]);
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
    dim3 block(N2/8);

    for (int i = 0; i < 5; i++) {
        Perf perf("safe_softmax_v2_f16x8_pack_f16");
        safe_softmax_v2_f16x8_pack_f16<N2/8><<<grid, block>>>(mat_A_device, mat_B_device, N2);
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