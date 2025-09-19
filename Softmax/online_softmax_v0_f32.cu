#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include <random>
#include <string>
#include "./include/util.hpp"

#define WARP_SIZE 32
// FP32
// DS required for Online Softmax
struct __align__(8) MD {
    float m;
    float d;
};

template <unsigned int NUM_THREADS>
__device__ __forceinline__ MD warp_reduce_md(MD val) {
#pragma unroll
    for (int delta = NUM_THREADS >> 1; delta >= 1; delta >>= 1) {
        MD other;
        other.m = __shfl_down_sync(0xffffffff, val.m, delta);
        other.d = __shfl_down_sync(0xffffffff, val.d, delta);

        MD bigger_MD = val.m >= other.m ? val : other;
        MD smaller_MD = val.m < other.m ? val : other;

        val.m = bigger_MD.m;
        val.d = bigger_MD.d + smaller_MD.d * __expf(smaller_MD.m - bigger_MD.m);
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
__global__ void online_softmax_v0_f32(float *mat_A, float *mat_B, int N) {
    float *thread_A_start = mat_A + blockIdx.x * N + threadIdx.x;
    float *thread_B_start = mat_B + blockIdx.x * N + threadIdx.x;
    float thread_A_val = thread_A_start[0];
    const int WARP_NUM = NUM_THREADS / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int landId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ MD warp_MD[WARP_NUM];
    // 默认1.0f, 代表假设每一个元素都是 max, e^{x-m} = 1.0f;
    MD val = {thread_A_val, 1.0f};
    MD warp_val = warp_reduce_md<WARP_SIZE>(val);
    if (landId == 0) warp_MD[warpId] = warp_val;
    __syncthreads();
    __shared__ MD block_val;
    if (warpId == 0) {
        val = warp_MD[landId];
        val = warp_reduce_md<WARP_NUM>(val);
        if (landId == 0) block_val = val;
    }
    __syncthreads();

    thread_B_start[0] = __fdividef(__expf(thread_A_val - block_val.m), block_val.d);
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

    cpu_safe_softmax(mat_A, mat_B_cpu_calc, N1, N2);

    float *mat_B_device = NULL;
    float *mat_B_gpu_calc = (float *)malloc(N1 * N2 * sizeof(float));
    cudaMalloc((void **)&mat_B_device, N1 * N2 * sizeof(float));
    dim3 grid(N1);
    dim3 block(N2);

    for (int i = 0; i < 5; i++) {
        Perf perf("online_softmax_v0_f32");
        online_softmax_v0_f32<N2><<<grid, block>>>(mat_A_device, mat_B_device, N2);
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