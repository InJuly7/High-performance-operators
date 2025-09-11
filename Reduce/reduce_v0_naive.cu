#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"
#define THREAD_PER_BLOCK 256

__global__ void reduce_v0_naive(float *vec_A, float *vec_B) {
    float *A_start = vec_A + blockIdx.x * blockDim.x;
    float *B_start = vec_B + blockIdx.x;
    __shared__ float SMem[THREAD_PER_BLOCK];
    SMem[threadIdx.x] = A_start[threadIdx.x];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((threadIdx.x & (2 * s - 1)) == 0) {
            // if (threadIdx.x % (2 * s) == 0) {
            SMem[threadIdx.x] += SMem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        B_start[0] = SMem[0];
    }
}

int main(int agrc, char **argv) {
    const int vector_size = 32 * 1024 * 1024;
    const int BLOCK_NUM = (vector_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
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

    for(int i = 0; i < 5; i++) {
        reduce_v0_naive<<<Grid, Block>>>(vector_device, vector_device_out);
        cudaDeviceSynchronize();
    }
    // reduce_v0_naive<<<Grid, Block>>>(vector_device, vector_device_out);
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
