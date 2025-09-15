#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"
#define THREAD_PER_BLOCK 1024
#define WARP_SIZE 32

template <unsigned int blockSize>
__device__ __forceinline__ float warp_shfl_Reduce(float sum) {
    if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);   // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);    // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);    // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);    // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v7_shuffle(float *vec_A, float *vec_B) {
    float *A_start = vec_A + blockIdx.x * blockDim.x * NUM_PER_THREAD;
    float *B_start = vec_B + blockIdx.x;
    float sum = 0;

#pragma unroll
    for (int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += A_start[iter * blockSize + threadIdx.x];
    }

    const int WARP_NUM = blockSize / WARP_SIZE;
    __shared__ float warpLevelSums[WARP_NUM];
    int warpId = threadIdx.x / WARP_SIZE;
    int landId = threadIdx.x & (WARP_SIZE - 1);
    
    sum = warp_shfl_Reduce<blockSize>(sum);
    // 每个 warp 中第一个 thread 存储 warp sum
    if (landId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();
    // 第一个 warp 再对所有的 warp sum 进行求和
    // 对第一个 warp sum 重新赋值
    if (warpId == 0) {
        sum = warpLevelSums[landId];
        sum = warp_shfl_Reduce<WARP_SIZE>(sum);
    }

    if (threadIdx.x == 0) B_start[0] = sum;
}

int main(int agrc, char **argv) {
    const int vector_size = 32 * 1024 * 1024;
    const int NUM_PER_THREAD = 4;
    const int BLOCK_NUM = (vector_size + THREAD_PER_BLOCK - 1) / (THREAD_PER_BLOCK * NUM_PER_THREAD);
    float *vector_host = (float *)malloc(vector_size * sizeof(float));
    generateRandomFloatArray(vector_host, vector_size);
    float *vector_device = NULL;
    cudaMalloc((void **)&vector_device, vector_size * sizeof(float));
    cudaMemcpy(vector_device, vector_host, vector_size * sizeof(float), cudaMemcpyHostToDevice);

    float *vector_host_out = (float *)malloc(BLOCK_NUM * sizeof(float));
    float *vector_device_out = NULL;
    cudaMalloc((void **)&vector_device_out, BLOCK_NUM * sizeof(float));

    float cpu_result = cpu_reduce(vector_host, vector_size);
    float gpu_result = 0.0;

    dim3 Grid(BLOCK_NUM);
    dim3 Block(THREAD_PER_BLOCK);
    for (int i = 0; i < 5; i++) {
        reduce_v7_shuffle<THREAD_PER_BLOCK, NUM_PER_THREAD><<<Grid, Block>>>(vector_device, vector_device_out);
        cudaDeviceSynchronize();
    }
    // reduce_v7_shuffle<THREAD_PER_BLOCK, NUM_PER_THREAD><<<Grid, Block>>>(vector_device, vector_device_out);
    // cudaDeviceSynchronize();
    cudaMemcpy(vector_host_out, vector_device_out, BLOCK_NUM * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_result = cpu_reduce(vector_host_out, BLOCK_NUM);

    std::cout << "cpu result: " << cpu_result << std::endl;
    std::cout << "gpu result: " << gpu_result << std::endl;

    compare_matrices(cpu_result, gpu_result);
    cudaFree(vector_device);
    cudaFree(vector_device_out);
    return 0;
}