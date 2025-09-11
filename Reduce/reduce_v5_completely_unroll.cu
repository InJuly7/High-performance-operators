#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"
#define THREAD_PER_BLOCK 256

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *cache, unsigned int tid) {
    if (blockSize >= 64) cache[tid] += cache[tid + 32];
    if (blockSize >= 32) cache[tid] += cache[tid + 16];
    if (blockSize >= 16) cache[tid] += cache[tid + 8];
    if (blockSize >= 8) cache[tid] += cache[tid + 4];
    if (blockSize >= 4) cache[tid] += cache[tid + 2];
    if (blockSize >= 2) cache[tid] += cache[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_v5_completely_unroll(float *vec_A, float *vec_B) {
    float *A_start = vec_A + blockIdx.x * blockDim.x * 2;
    float *B_start = vec_B + blockIdx.x;
    __shared__ float SMem[blockSize];
    SMem[threadIdx.x] = A_start[threadIdx.x] + A_start[threadIdx.x + blockDim.x];
    __syncthreads();

    if (blockSize >= 512) {
        SMem[threadIdx.x] += SMem[threadIdx.x + 256];
        __syncthreads();
    }
    if (blockSize >= 256 && threadIdx.x < 128) {
        SMem[threadIdx.x] += SMem[threadIdx.x + 128];
        __syncthreads();
    }
    if (blockSize >= 128 && threadIdx.x < 64) {
        SMem[threadIdx.x] += SMem[threadIdx.x + 64];
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce<blockSize>(SMem);
    if (threadIdx.x == 0) B_start[0] = SMem[0];
}

int main(int agrc, char **argv) {
    const int vector_size = 32 * 1024 * 1024;
    const int BLOCK_NUM = (vector_size + THREAD_PER_BLOCK - 1) / (THREAD_PER_BLOCK * 2);
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

    reduce_v5_completely_unroll<THREAD_PER_BLOCK><<<Grid, Block>>>(vector_device, vector_device_out);
    cudaDeviceSynchronize();
    cudaMemcpy(vector_host_out, vector_device_out, BLOCK_NUM * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_result = cpu_reduce(vector_host_out, BLOCK_NUM);

    std::cout << "cpu result: " << cpu_result << std::endl;
    std::cout << "gpu result: " << gpu_result << std::endl;

    compare_matrices(cpu_result, gpu_result);
    cudaFree(vector_device);
    cudaFree(vector_device_out);
    return 0;
}