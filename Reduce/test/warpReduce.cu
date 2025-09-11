#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>

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

__device__ void warpReduce(volatile float *cache, unsigned int tid) {
    cache[tid] += cache[tid + 32];
    // __syncwarp();
    cache[tid] += cache[tid + 16];  // if(tid < 16)
    // __syncwarp();
    cache[tid] += cache[tid + 8];  // if(tid < 8)
    // __syncwarp();
    cache[tid] += cache[tid + 4];
    // __syncwarp();
    cache[tid] += cache[tid + 2];
    // __syncwarp();
    cache[tid] += cache[tid + 1];
}

__global__ void Reduce0(float *d_in, float *d_out) {
    __shared__ float sdata[64];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) d_out[0] = sdata[0];
}

__global__ void Reduce1(float *d_in, float *d_out) {
    __shared__ float sdata[64];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        __syncwarp();
        sdata[tid] += sdata[tid + 16];  // if(tid < 16)
        __syncwarp();
        sdata[tid] += sdata[tid + 8];  // if(tid < 8)
        __syncwarp();
        sdata[tid] += sdata[tid + 4];
        __syncwarp();
        sdata[tid] += sdata[tid + 2];
        __syncwarp();
        sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) d_out[0] = sdata[0];
}

std::vector<float> generateRandomVector(int N) {
    std::vector<float> vec(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

float reduce(float *h_a, int N) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        result += h_a[i];
    }
    return result;
}

int main(int agrc, char **argv) {
    int N = 1024;
    std::vector<float> h_a = generateRandomVector(N);
    float *d_a = NULL;
    checkCudaErr(cudaMalloc((void **)&d_a, N * sizeof(float)));
    float cpu_result = reduce(h_a.data(), N);
    float gpu_result = 0.0;
    float *d_c = NULL;
    std::vector<float> h_c(1);
    checkCudaErr(cudaMalloc((void **)&d_c, 1 * sizeof(float)));
    checkCudaErr(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 Grid(1, 1);
    dim3 Block(64, 1);
    if (std::string(argv[1]) == "reduce0")
        Reduce0<<<Grid, Block>>>(d_a, d_c);
    else if (std::string(argv[1]) == "reduce1")
        Reduce1<<<Grid, Block>>>(d_a, d_c);
    checkCudaErr(cudaGetLastError());
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(h_c.data(), d_c, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    gpu_result = h_c[0];
    std::cout << "cpu result: " << cpu_result << std::endl;
    std::cout << "gpu result: " << gpu_result << std::endl;
    if (abs(gpu_result - cpu_result) / cpu_result > 1e-3) std::cout << "Gpu Result Error!!\n";
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}