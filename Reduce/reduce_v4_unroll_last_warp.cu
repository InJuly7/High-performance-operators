#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "./include/util.hpp"
#define THREAD_PER_BLOCK 256

// volatile 告诉编译器不要对被修饰的变量进行优化，每次访问都必须从内存中重新读取
__device__ void warpReduce4(volatile float *cache) {
    cache[threadIdx.x] += cache[threadIdx.x + 32];
    cache[threadIdx.x] += cache[threadIdx.x + 16];  // if(tid < 16)
    cache[threadIdx.x] += cache[threadIdx.x + 8];   // if(tid < 8)
    cache[threadIdx.x] += cache[threadIdx.x + 4];
    cache[threadIdx.x] += cache[threadIdx.x + 2];
    cache[threadIdx.x] += cache[threadIdx.x + 1];
}

__global__ void reduce_v4_unroll_last_warp(float *vec_A, float *vec_B) {
    float *A_start = vec_A + blockIdx.x * blockDim.x * 2;
    float *B_start = vec_B + blockIdx.x;
    __shared__ float SMem[THREAD_PER_BLOCK];
    SMem[threadIdx.x] = A_start[threadIdx.x] + A_start[threadIdx.x + blockDim.x];
    // warp粒度同步, 当前warp读取的值,可能别的warp还未写入
    // 写后读冲突 SMem[128] = SMem[128] + SMem[256] --> SMem[0] = SMem[0] + SMem[128]
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            SMem[threadIdx.x] += SMem[threadIdx.x + s];
        }
        // 写后读冲突 SMem[64] = SMem[64] + SMem[192] --> SMem[0] = SMem[0] + SMem[64]
        __syncthreads();
    }
    // 最后一个 warp 手动展开, 避免 warp divergence
    if (threadIdx.x < 32) warpReduce4(SMem);
    // if (tid < 32) {
    //     SMem[tid]+=SMem[tid+32];
    //     __syncwarp();
    //     SMem[tid]+=SMem[tid+16]; // if(tid < 16)
    //     __syncwarp();
    //     SMem[tid]+=SMem[tid+8];  // if(tid < 8)
    //     __syncwarp();
    //     SMem[tid]+=SMem[tid+4];
    //     __syncwarp();
    //     SMem[tid]+=SMem[tid+2];
    //     __syncwarp();
    //     SMem[tid]+=SMem[tid+1];
    // }
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

    reduce_v4_unroll_last_warp<<<Grid, Block>>>(vector_device, vector_device_out);
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