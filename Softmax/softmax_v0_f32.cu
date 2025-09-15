#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ float warpsum[NUM_WARPS];
    val = warp_reduce_sum_f32(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    // tid == 0 返回 block_reduce_sum 
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : 0.0f;
        val = warp_reduce_sum_f32(val);
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
__global__ void softmax_v0_f32(float *mat_A, float *mat_B, int N) {
    float *thread_A_start = mat_A + blockIdx.x * N + threadIdx.x;
    float *thread_B_start = mat_B + blockIdx.x * N + threadIdx.x;
    __shared__ float exp_sum;
    float exp_val = expf(thread_A_start[0]);
    float local_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if(threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();
    thread_B_start[0] = exp_val / exp_sum;
}

int main() {
    const int N1 = 4096;
    const int N2 = 1024;
    float *mat_A = (float *)malloc(N1 * N2 * sizeof(float));
    float *mat_B_cpu_calc = (float *)malloc(N1 * N2 * sizeof(float));
    generateRandomFloatArray(mat_A, N1 * N2);
    float *mat_A_device = NULL;
    cudaMalloc((void **)&mat_A_device, N1 * N2 * sizeof(float));
    cudaMemcpy(mat_A_device, mat_A, N1 * N2 * sizeof(float), cudaMemcpyHostToDevice);

    cpu_softmax(mat_A, mat_B_cpu_calc, N1, N2);

    float *mat_B_device = NULL;
    float *mat_B_gpu_calc = (float *)malloc(N1 * N2 * sizeof(float));
    cudaMalloc((void **)&mat_B_device, N1 * N2 * sizeof(float));
    dim3 grid(N1);
    dim3 block(N2);

    for (int i = 0; i < 5; i++) {
        Perf perf("softmax_v0_f32");
        softmax_v0_f32<N2><<<grid, block>>>(mat_A_device, mat_B_device, N2);
    }

    cudaMemcpy(mat_B_gpu_calc, mat_B_device, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost);
    printFloatArray(mat_B_cpu_calc, 10);
    printFloatArray(mat_B_gpu_calc, 10);
    compare_matrices(N1, N2, mat_B_cpu_calc, mat_B_gpu_calc);

    free(mat_A);
    free(mat_B_cpu_calc);
    free(mat_B_gpu_calc);
    cudaFree(mat_A_device);
    cudaFree(mat_B_device);
}