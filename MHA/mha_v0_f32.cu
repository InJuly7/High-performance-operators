#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./include/util.hpp"

#define WARP_SIZE 32

// S = (Q @ K^T) * rsqrtf(dk)
__global__ void qk_matmul(float *Q, float *K, float *S, int H, int S1, int S2, int dk) {
    Q += blockIdx.y * S1 * dk + blockIdx.x * dk;
    K += blockIdx.y * S2 * dk;
    S += blockIdx.y * S1 * S2 + blockIdx.x * S2;
    float scale = rsqrtf(dk);

    float temp = 0.0f;
    for (int j = 0; j < dk; j++) {
        temp += Q[j] * K[threadIdx.x * dk + j];
    }
    S[threadIdx.x] = temp * scale;
}

// P = softmax(S)
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ float warpsum[NUM_WARPS];
    val = warp_reduce_sum_f32(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        val = 0;
        for (int i = 0; i < NUM_WARPS; i++) {
            val += warpsum[i];
        }
        return val;
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ float block_reduce_max_f32(float val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ float warpmax[NUM_WARPS];
    val = warp_reduce_max_f32(val);
    if (laneId == 0) warpmax[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        val = -FLT_MAX;
        for (int i = 0; i < NUM_WARPS; i++) {
            val = fmax(val, warpmax[i]);
        }
        return val;
    }
    return val;
}

template <unsigned int NUM_THREADS>
__global__ void safe_softmax(float *S, float *P, int H, int S1, int S2) {
    float *thread_A_start = S + blockIdx.y * S1 * S2 + blockIdx.x * S2 + threadIdx.x;
    float *thread_B_start = P + blockIdx.y * S1 * S2 + blockIdx.x * S2 + threadIdx.x;

    __shared__ float exp_sum;
    __shared__ float global_max;
    float val = thread_A_start[0];
    float local_max = block_reduce_max_f32<NUM_THREADS>(val);
    if (threadIdx.x == 0) global_max = local_max;
    __syncthreads();
    float exp_val = expf(val - global_max);
    float local_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();
    thread_B_start[0] = exp_val / exp_sum;
}

// O = P @ V
__global__ void pv_matmul(float *P, float *V, float* O, int H, int S1, int S2, int dk) {
    P += blockIdx.y * S1 * S2 + blockIdx.x * S2;
    V += blockIdx.y * S2 * dk;
    O += blockIdx.y * S1 * dk + blockIdx.x * dk; 
    float temp = 0.0f;
    for(int i = 0; i < S2; i++) {
        temp += P[i] * V[i * dk + threadIdx.x];
    }
    O[threadIdx.x] = temp;
}

int main() {
    const int S1 = 2048;
    const int S2 = 1024;
    const int dk = 128;
    const int H = 8;

    float *Q = (float *)malloc(H * S1 * dk * sizeof(float));
    float *K = (float *)malloc(H * S2 * dk * sizeof(float));
    float *V = (float *)malloc(H * S2 * dk * sizeof(float));
    float *O_gpu_cal = (float *)malloc(H * S1 * dk * sizeof(float));
    float *O_cpu_cal = (float *)malloc(H * S1 * dk * sizeof(float));

    generateRandomFloatArray(Q, H * S1 * dk);
    generateRandomFloatArray(K, H * S2 * dk);
    generateRandomFloatArray(V, H * S2 * dk);

    float *d_Q, *d_K, *d_V;
    cudaMalloc((void **)&d_Q, H * S1 * dk * sizeof(float));
    cudaMalloc((void **)&d_K, H * S2 * dk * sizeof(float));
    cudaMalloc((void **)&d_V, H * S2 * dk * sizeof(float));

    cudaMemcpy(d_Q, Q, H * S1 * dk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, H * S2 * dk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, H * S2 * dk * sizeof(float), cudaMemcpyHostToDevice);

    float *d_S = NULL;
    cudaMalloc((void **)&d_S, H * S1 * S2 * sizeof(float));
    float *d_P = NULL;
    cudaMalloc((void **)&d_P, H * S1 * S2 * sizeof(float));
    float *d_O = NULL;
    cudaMalloc((void **)&d_O, H * S1 * dk * sizeof(float));
    // S = (Q @ K^T) * rsqrtf(dk)    
    dim3 grid_1 = {S1, H, 1};
    dim3 block_1 = {S2};
    qk_matmul<<<grid_1, block_1>>>(d_Q, d_K, d_S, H, S1, S2, dk);
    cudaDeviceSynchronize();

    // float *S_cpu_cal = (float *)malloc(H * S1 * S2 * sizeof(float));
    // float *S_gpu_cal = (float *)malloc(H * S1 * S2 * sizeof(float));
    // cpu_qk_matmul(Q, K, S_cpu_cal, H, S1, S2, dk);
    // cudaMemcpy(S_gpu_cal, d_S, H * S1 * S2 * sizeof(float), cudaMemcpyDeviceToHost);
    // printFloatArray(S_cpu_cal, 10);
    // printFloatArray(S_gpu_cal, 10);
    // compare_matrices(H, S1, S2, S_cpu_cal, S_gpu_cal);

    // P = softmax(S)
    dim3 grid_2 = {S1, H, 1};
    dim3 block_2 = {S2};
    safe_softmax<S2><<<grid_2, block_2>>>(d_S, d_P, H, S1, S2);
    cudaDeviceSynchronize();

    // float *S_cpu_cal = (float *)malloc(H * S1 * S2 * sizeof(float));
    // float *P_cpu_cal = (float *)malloc(H * S1 * S2 * sizeof(float));
    // float *P_gpu_cal = (float *)malloc(H * S1 * S2 * sizeof(float));
    // cpu_qk_matmul(Q, K, S_cpu_cal, H, S1, S2, dk);
    // cpu_safe_softmax(S_cpu_cal, P_cpu_cal, H, S1, S2);
    // cudaMemcpy(P_gpu_cal, d_P, H * S1 * S2 * sizeof(float), cudaMemcpyDeviceToHost);
    // printFloatArray(P_cpu_cal, 10);
    // printFloatArray(P_gpu_cal, 10);
    // compare_matrices(H, S1, S2, P_cpu_cal, P_gpu_cal);

    // O = P @ V
    dim3 grid_3 = {S1, H, 1};
    dim3 block_3 = {dk};
    pv_matmul<<<grid_3, block_3>>>(d_P, d_V, d_O, H, S1, S2, dk);
    cudaDeviceSynchronize();
    cudaMemcpy(O_gpu_cal, d_O, H * S1 * dk * sizeof(float), cudaMemcpyDeviceToHost);

    float *S_cpu_cal = (float *)malloc(H * S1 * S2 * sizeof(float));
    float *P_cpu_cal = (float *)malloc(H * S1 * S2 * sizeof(float));
    cpu_qk_matmul(Q, K, S_cpu_cal, H, S1, S2, dk);
    cpu_safe_softmax(S_cpu_cal, P_cpu_cal, H, S1, S2);
    cpu_pv_matmul(P_cpu_cal, V, O_cpu_cal, H, S1, S2, dk);

    printFloatArray(O_cpu_cal, 10);
    printFloatArray(O_gpu_cal, 10);
    compare_matrices(H, S1, dk, O_cpu_cal, O_gpu_cal);

    free(Q);
    free(K);
    free(V);
    free(O_gpu_cal);
    free(O_cpu_cal);
    free(S_cpu_cal);
    free(P_cpu_cal);


    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_S);
    cudaFree(d_P);
}