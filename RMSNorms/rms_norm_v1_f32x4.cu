#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


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
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : 0.0f;
        val = warp_reduce_sum_f32(val);
    }
    return val;
}

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <unsigned int NUM_THREADS>
__global__ void rms_norm_v1_f32x4(float *mat_A, float *mat_B, float g, int N, int K) {
    float *mat_A_start = mat_A + blockIdx.x * K;
    float *mat_B_start = mat_B + blockIdx.x * K;
    const float epsilon = 1e-5f;
    float4 reg_A = FLOAT4(mat_A_start[threadIdx.x * 4]);
    // 块内共享, 求出当前行 rsqrtf(sum(ai^2)/K)
    __shared__ float s_variance;
    float variance = reg_A.x * reg_A.x + reg_A.y * reg_A.y + reg_A.z * reg_A.z + reg_A.w * reg_A.w;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();

    float4 reg_B;
    reg_B.x = reg_A.x * s_variance * g;
    reg_B.y = reg_A.y * s_variance * g;
    reg_B.z = reg_A.z * s_variance * g;
    reg_B.w = reg_A.w * s_variance * g;
    FLOAT4(mat_B_start[threadIdx.x * 4]) = reg_B;
}

int main() {
    const int N = 4096;
    const int K = 1024;
    const float g = 0.35;

    float *mat_A = (float *)malloc(N * K * sizeof(float));
    float *mat_B_cpu_calc = (float *)malloc(N * K * sizeof(float));
    generateRandomFloatArray(mat_A, N * K);
    float *mat_A_device = NULL;
    cudaMalloc((void **)&mat_A_device, N * K * sizeof(float));
    cudaMemcpy(mat_A_device, mat_A, N * K * sizeof(float), cudaMemcpyHostToDevice);

    cpu_rms_norm(mat_A, mat_B_cpu_calc, g, N, K);

    float *mat_B_device = NULL;
    float *mat_B_gpu_calc = (float *)malloc(N * K * sizeof(float));
    cudaMalloc((void **)&mat_B_device, N * K * sizeof(float));
    dim3 grid(N);
    dim3 block(K/4);

    for (int i = 0; i < 5; i++) {
        Perf perf("rms_norm_v1_f32x4");
        rms_norm_v1_f32x4<K><<<grid, block>>>(mat_A_device, mat_B_device, g, N, K);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(mat_B_gpu_calc, mat_B_device, N * K * sizeof(float), cudaMemcpyDeviceToHost);
    printFloatArray(mat_B_cpu_calc, 10);
    printFloatArray(mat_B_gpu_calc, 10);
    compare_matrices(N, K, mat_B_cpu_calc, mat_B_gpu_calc);

    free(mat_A);
    free(mat_B_cpu_calc);
    free(mat_B_gpu_calc);
    cudaFree(mat_A_device);
    cudaFree(mat_B_device);
}