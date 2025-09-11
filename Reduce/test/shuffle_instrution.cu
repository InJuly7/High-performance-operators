#include <cuda_runtime.h>
#include <stdio.h>

// 辅助函数:检查错误
#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

// 简单的向量加法 kernel,展示 shuffle 的使用
__global__ void vectorAddWithShuffle(float *a, float *b, float *c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 每个线程加载并计算一个元素
    float sum = 0;
    if (tid < N) sum = a[tid] + b[tid];
    c[tid] = sum;  // 存储原始结果
    // 使用 shuffle 在 warp 内交换数据的演示
    // 每个线程会打印自己的值和后面第4个线程的值
    float shuffled = __shfl_down_sync(0xffffffff, sum, 4);
    if (tid < N - 4) {
        // 打印当前线程的值和通过shuffle获得的值
        printf("Thread %d: My value = %.1f, Value from 4 threads ahead = %.1f\n", tid, sum, shuffled);
    }
}

int main() {
    const int N = 32;  // 使用一个 warp 大小的数据做演示
    const int bytes = N * sizeof(float);

    // 分配主机内存
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // 复制数据到设备
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 启动 kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddWithShuffle<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // 复制结果回主机
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // 验证结果
    printf("\nResults:\n");
    for (int i = 0; i < N; i++) {
        printf("c[%d] = %.1f\n", i, h_c[i]);
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return 0;
}
