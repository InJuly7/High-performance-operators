#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include "./include/util.hpp"

__global__ void sgemm_transpose() {
    
}

__global__ void sgemm() {

}
__global__ void softmax() {

}

int main()
{
    const int N1 = 8192;
    const int N2 = 8192;
    const int dk = 64;
    const int H = 8;

    float *Q = (float *)malloc(H * N1 * dk * sizeof(float));
    float *K = (float *)malloc(H * N2 * dk * sizeof(float));
    float *V = (float *)malloc(H * N2 * dk * sizeof(float));
    float *O_gpu_cal = (float *)malloc(H * N1 * dk * sizeof(float));
    float *O_cpu_cal = (float *)malloc(H * N1 * dk * sizeof(float));

    generateRandomFloatArray(Q, H * N1 * dk);
    generateRandomFloatArray(K, H * N2 * dk);
    generateRandomFloatArray(V, H * N2 * dk);

    cpu_multihead_attention(Q, K, V, O_cpu_cal, N1, N2, dk, H);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc((void **)&d_Q, H * N1 * dk * sizeof(float));
    cudaMalloc((void **)&d_K, H * N2 * dk * sizeof(float));
    cudaMalloc((void **)&d_V, H * N2 * dk * sizeof(float));
    cudaMalloc((void **)&d_O, H * N1 * dk * sizeof(float));

    cudaMemcpy(d_Q, Q, H * N1 * dk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, H * N2 * dk * sizeof(float), cudaMemcpyHostToDevice);

    // S = (Q @ K^T) * rsqrtf(dk)
    float *d_S = NULL;
    cudaMalloc((void **)&d_S, H * N1 * N2 * sizeof(float));
    dim3 grid = {};
    dim3 block = {};
    sgemm_transpose<<<grid, block>>>(d_Q, d_K, d_S, H, N1, N2, dk);
    cudaDeviceSynchronize();

    // P = softmax(S)
    float *d_P = NULL;
    cudaMalloc((void **)d_P, H * N1 * N2 * sizeof(float));
    dim3 grid = {};
    dim3 block = {};
    softmax<<<grid, block>>>(d_S, d_P, N2);
    cudaDeviceSynchronize();

    // O = P @ V
    sgemm<<<grid, block>>>(d_P, d_V, d_O, H, N1, N2, dk);
    cudaDeviceSynchronize();
    cudaMemcpy(O_gpu_cal, d_O, H * N1 * dk * sizeof(float), cudaMemcpyDeviceToHost);

    printFloatArray(O_cpu_cal, 10);
    printFloatArray(O_gpu_cal, 10);
    compare_matrices(O_cpu_cal, O_gpu_cal);
    
    free(Q);
    free(K);
    free(V);
    free(O_gpu_cal);
    free(O_cpu_cal);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_S);
    cudaFree(d_P);
}