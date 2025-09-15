#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]
// FP32
// DS required for Online Softmax
struct __align__(8) MD {
    float m;
    float d;
};

template<unsigned int NUM_THREADS>
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
__global__ void online_softmax_v1_f32x4(float *mat_A, float *mat_B, int N) {
    float *thread_A_start = mat_A + blockIdx.x * N + 4 * threadIdx.x;
    float *thread_B_start = mat_B + blockIdx.x * N + 4 * threadIdx.x;
    const int WARP_NUM = NUM_THREADS / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int landId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ MD warp_MD[WARP_NUM];
    float4 reg_A = FLOAT4(thread_A_start[0]);

    float local_m = fmaxf(fmaxf((reg_A).x, (reg_A).y), fmaxf((reg_A).z, (reg_A).w));
    float local_d = __expf(reg_A.x - local_m) + __expf(reg_A.y - local_m) + __expf(reg_A.z - local_m) + __expf(reg_A.w - local_m);
    MD local_md = {local_m, local_d};
    MD warp_md = warp_reduce_md<WARP_SIZE>(local_md);
    if(landId == 0) warp_MD[warpId] = warp_md;
    __syncthreads();
    
    __shared__ MD block_md;
    MD zero_md = {-FLT_MAX, 1.0f};
    if(warpId == 0) {
        local_md = (landId < WARP_NUM) ? warp_MD[landId] : zero_md;
        warp_md = warp_reduce_md<WARP_NUM>(local_md);
        if(landId == 0) block_md = warp_md;
    }
    __syncthreads();
    float4 reg_B;
    reg_B.x = __fdividef(__expf(reg_A.x - block_md.m), block_md.d);
    reg_B.y = __fdividef(__expf(reg_A.y - block_md.m), block_md.d);
    reg_B.z = __fdividef(__expf(reg_A.z - block_md.m), block_md.d);
    reg_B.w = __fdividef(__expf(reg_A.w - block_md.m), block_md.d);
    FLOAT4(thread_B_start[0]) = reg_B;
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
    dim3 block(N2/4);

    for (int i = 0; i < 5; i++) {
        Perf perf("online_softmax_v1_f32x4");
        online_softmax_v1_f32x4<N2/4><<<grid, block>>>(mat_A_device, mat_B_device, N2);
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