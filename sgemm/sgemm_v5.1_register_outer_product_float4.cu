#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./include/util.hpp"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

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

// 每个线程 计算 ROW_FRAG * COL_FRAG 个元素
template <unsigned int BM, unsigned int BK, unsigned int BN, unsigned int ROW_FRAG, unsigned int COL_FRAG, unsigned int M, unsigned int K,
          unsigned int N>
__global__ void cuda_sgemm(float *matrix_A_device, float *matrix_B_device, float *matrix_C_device) {
    float *A_ptr_start = matrix_A_device + blockIdx.y * BM * K;
    float *B_ptr_start = matrix_B_device + blockIdx.x * BN;
    float *C_ptr_start = matrix_C_device + blockIdx.y * BM * N + blockIdx.x * BN;

    __shared__ float A_smem[BM][BK];
    __shared__ float B_smem[BK][BN];
    const int NUM_PER_THREAD = ROW_FRAG * COL_FRAG;
    int thread_start = threadIdx.x * NUM_PER_THREAD;
    int A_smem_y = thread_start / BK, A_smem_x = thread_start % BK;
    int B_smem_y = thread_start / BN, B_smem_x = thread_start % BN;

    float res[ROW_FRAG][COL_FRAG] = {0.0f};
    float row_frag[ROW_FRAG] = {0.0f};
    float col_frag[COL_FRAG] = {0.0f};

    const int c_row_start = ((threadIdx.x * COL_FRAG) / BN) * ROW_FRAG;
    const int c_col_start = (threadIdx.x * COL_FRAG) % BN;

    for (int i = 0; i < K; i += BK) {
        // GM ==> SM
        // 将相同内存地址的数据重新解释为一个 float4 对象
        for(int j = 0; j < NUM_PER_THREAD; j+=4) {
            FETCH_FLOAT4(A_smem[A_smem_y][A_smem_x + j]) = FETCH_FLOAT4(A_ptr_start[A_smem_y * K + A_smem_x + i + j]);
            FETCH_FLOAT4(B_smem[B_smem_y][B_smem_x + j]) = FETCH_FLOAT4(B_ptr_start[(i + B_smem_y) * N + B_smem_x + j]);
        }
        __syncthreads();

        
        for (int j = 0; j < BK; j++) {
            for (int j1 = 0; j1 < ROW_FRAG; j1++) row_frag[j1] = A_smem[c_row_start + j1][j];
            FETCH_FLOAT4(col_frag[0]) = FETCH_FLOAT4(B_smem[j][c_col_start]);
            for (int k1 = 0; k1 < ROW_FRAG; k1++) {
                for (int k2 = 0; k2 < COL_FRAG; k2++) {
                    res[k1][k2] += row_frag[k1] * col_frag[k2];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < ROW_FRAG; i++) {
        for (int j = 0; j < COL_FRAG; j++) {
            C_ptr_start[(c_row_start + i) * N + c_col_start + j] = res[i][j];
        }
    }
}

int main() {
    const int m = 1024, k = 1024, n = 1024;
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

    const int bm = 64;
    const int bk = 64;
    const int bn = 64;
    const int BLOCK_SIZE = 256;
    const int ROW_FRAG = 4;
    const int COL_FRAG = 4;
    dim3 block(BLOCK_SIZE);
    dim3 grid(n / bn, m / bm);
    cuda_sgemm<bm, bk, bn, ROW_FRAG, COL_FRAG, m, k, n><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device);

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
