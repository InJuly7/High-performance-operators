#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>

#define THREAD_PER_BLOCK 256
static int checkCudaError(cudaError_t code, const char *expr, const char *file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    // 偶数号block的首地址
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    // unrolling 2 data blocks
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[ti] += idata[tid + stride];
        }d
        // synchronize within threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)  // 注意此处的stride循环条件发生了变化(相较于上面的函数)
    {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    /*
    上面这段代码可以看成
    for (int stride = 32; stride > 0; stride >>= 1)// 注意此处的stride循环条件发生了变化(相较于上面的函数)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
    实际上就是把上面的循环展开了
    */
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}