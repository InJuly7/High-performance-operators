#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./include/util.hpp"

void cpu_sgemm(float *matrix_A_host, float *matrix_B_host, float *matrix_C_host_cpu_calc, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += matrix_A_host[m * K + k] * matrix_B_host[k * N + n];
            }
            matrix_C_host_cpu_calc[m * N + n] = temp;
        }
    }
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
template <unsigned int BM, unsigned int BK, unsigned int BN, unsigned int NUM_PER_THREAD, unsigned int M, unsigned int K, unsigned int N>
__global__ void cuda_sgemm(float *matrix_A_device, float *matrix_B_device, float *matrix_C_device) {
    float *A_ptr_start = matrix_A_device + blockIdx.y * BM * K;
    float *B_ptr_start = matrix_B_device + blockIdx.x * BN;
    float *C_ptr_start = matrix_C_device + blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ float A_smem[BM][BK];
    __shared__ float B_smem[BK][BN];

    int A_smem_y = (threadIdx.x * NUM_PER_THREAD) / BK;
    int A_smem_x = (threadIdx.x * NUM_PER_THREAD) % BK;
    int B_smem_y = (threadIdx.x * NUM_PER_THREAD) / BN;
    int B_smem_x = (threadIdx.x * NUM_PER_THREAD) % BN;

    float temp[NUM_PER_THREAD] = {0.0f};

    for(int i = 0; i < K; i += BK) {
        // GM ==> SM
        // 将相同内存地址的数据重新解释为一个 float4 对象
        FETCH_FLOAT4(A_smem[A_smem_y][A_smem_x]) =
            FETCH_FLOAT4(A_ptr_start[A_smem_y * K + A_smem_x + i]);
        FETCH_FLOAT4(B_smem[B_smem_y][B_smem_x]) =
            FETCH_FLOAT4(B_ptr_start[(i + B_smem_y) * N + B_smem_x]);

        __syncthreads();

        for (int j = 0; j < NUM_PER_THREAD; j++) {
            for (int k = 0; k < BK; k++) {
                temp[j] += A_smem[A_smem_y][k] * B_smem[k][j + B_smem_x];
            }
        }
        __syncthreads();
    }
    int C_gmem_y = (threadIdx.x * NUM_PER_THREAD) / BN;
    int C_gmem_x = (threadIdx.x * NUM_PER_THREAD) % BN;
    for(int i = 0; i < NUM_PER_THREAD; i++) {
        C_ptr_start[C_gmem_y * N + C_gmem_x + i] = temp[i];
    }
}

int main() {
    const int m = 256, k = 256, n = 256;
    const int mem_size_A = m * k * sizeof(float);
    const int mem_size_B = k * n * sizeof(float);
    const int mem_size_C = m * n * sizeof(float);

    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);

    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    generateRandomFloatArray(matrix_A_host, m * k);
    generateRandomFloatArray(matrix_B_host, k * n);

    memset(matrix_C_host_cpu_calc, 0, mem_size_C);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);

    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);

    printFloatArray(matrix_C_host_cpu_calc, 10);

    const int bm = 32;
    const int bk = 32;
    const int bn = 32;
    const int BLOCK_SIZE = 256;
    const int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SIZE);
    dim3 grid(n / bn, m / bm);
    cuda_sgemm<bm, bk, bn, NUM_PER_THREAD, m, k, n><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    printFloatArray(matrix_C_host_gpu_calc, 10);

    compare_matrices(m * n, matrix_C_host_cpu_calc, matrix_C_host_gpu_calc);

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu_calc);
    free(matrix_C_host_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
}
