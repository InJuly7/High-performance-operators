#include <cuda_runtime.h>
#include <stdio.h>

// 使用循环展开的核函数
__global__ void kernelWithUnroll(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float result = 0.0f;
#pragma unroll
        for (int i = 0; i < 128; i++) {
            result += sinf(result + i);
        }

        output[idx] = result;
    }
}

__global__ void kernelWithoutUnroll(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float result = 0.0f;
        for (int i = 0; i < 128; i++) {
            result += sinf(result + i);
        }

        output[idx] = result;
    }
}

int main() {
    const int N = 1 << 20;   // 1M elements
    const int REPEAT = 100;  // 重复100次测试

    // 分配内存
    float *d_output1, *d_output2;
    cudaMalloc(&d_output1, N * sizeof(float));
    cudaMalloc(&d_output2, N * sizeof(float));

    // 设置线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // warming up
    kernelWithUnroll<<<blocksPerGrid, threadsPerBlock>>>(d_output1, N);
    cudaDeviceSynchronize();

    // 创建CUDA Event用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 测试使用循环展开的版本
    float unrollTime = 0.0f;
    cudaEventRecord(start);

    for (int i = 0; i < REPEAT; i++) {
        kernelWithUnroll<<<blocksPerGrid, threadsPerBlock>>>(d_output1, N);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&unrollTime, start, stop);

    // 测试不使用循环展开的版本
    float noUnrollTime = 0.0f;
    cudaEventRecord(start);

    for (int i = 0; i < REPEAT; i++) {
        kernelWithoutUnroll<<<blocksPerGrid, threadsPerBlock>>>(d_output2, N);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&noUnrollTime, start, stop);

    // 输出结果
    printf("Time with unroll: %.3f ms\n", unrollTime);
    printf("Time without unroll: %.3f ms\n", noUnrollTime);
    printf("Performance improvement: %.2f%%\n", (noUnrollTime - unrollTime) * 100.0f / noUnrollTime);

    // 清理
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
