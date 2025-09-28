#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./include/util.hpp"

#define WARP_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val)))[0]
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val)))[0]

template <unsigned int sram_size>
__global__ void flash_attn_v2(float *Q, float *K, float *V, float *l, float *O, int S, int dk, int Tc, int Tr, int Bc, int Br) {
    float scale = rsqrtf(dk);
    int tile_size = Bc * dk;
    __shared__ float SMem[sram_size];
    float *SMem_Q = &SMem[0];
    float *SMem_K = &SMem[tile_size];
    float *SMem_V = &SMem[tile_size * 2];
    float *SMem_S = &SMem[tile_size * 3];

    Q += blockIdx.x * S * dk + blockIdx.y * Br * dk;
    K += blockIdx.x * S * dk;
    V += blockIdx.x * S * dk;
    O += blockIdx.x * S * dk + blockIdx.y * Br * dk;
    l += blockIdx.x * S + blockIdx.y * Br;

    int row_Q = blockIdx.y * Br + threadIdx.x;
    // Load GMemQ To SMemQ
    for (int i = 0; i < dk; i += 4) {
        FLOAT4(SMem_Q[threadIdx.x * dk + i]) = FLOAT4(Q[threadIdx.x * dk + i]);
    }
    
    float row_m_prev = -FLT_MAX;
    float row_l_prev = 0;

    for (int tc = 0; tc <= blockIdx.y; tc++) {
        for (int i = 0; i < dk; i += 4) {
            FLOAT4(SMem_K[threadIdx.x * dk + i]) = FLOAT4(K[tc * tile_size + threadIdx.x * dk + i]);
            FLOAT4(SMem_V[threadIdx.x * dk + i]) = FLOAT4(V[tc * tile_size + threadIdx.x * dk + i]);
        }
        __syncthreads();
        float row_m = -FLT_MAX;
        // causal mask
        for (int bc = 0; bc < Bc; bc++) {
            // causal mask
            if (tc * Bc + bc > row_Q) break;
            float temp = 0.0f;
            for (int i = 0; i < dk; i++) {
                temp += SMem_Q[threadIdx.x * dk + i] * SMem_K[bc * dk + i];
            }
            temp *= scale;
            SMem_S[threadIdx.x * Bc + bc] = temp;
            row_m = fmax(row_m, temp);
        }
        float row_m_new = fmax(row_m, row_m_prev);

        float row_l = 0.0f;
        for(int bc = 0; bc < Bc; bc++) {
            // causal mask
            if(tc * Bc + bc > row_Q) break;
            SMem_S[threadIdx.x * Bc + bc] = __expf(SMem_S[threadIdx.x * Bc + bc] - row_m_new);
            row_l += SMem_S[threadIdx.x * Bc + bc];
        }
        float row_l_new = row_l + row_l_prev * __expf(row_m_prev - row_m_new);

        for(int i = 0; i < dk; i++) {
            float pv = 0.0f;
            for(int bc = 0; bc < Bc; bc++) {
                // causal mask
                if (tc * Bc + bc > row_Q) break;
                pv += SMem_S[threadIdx.x * Bc + bc] * SMem_V[bc * dk + i];
            }
            O[threadIdx.x * dk + i] = O[threadIdx.x * dk + i] * (row_l_prev / row_l_new) * __expf(row_m_prev - row_m_new) + pv / row_l_new;
        }
        row_m_prev = row_m_new;
        row_l_prev = row_l_new;
        __syncthreads(); 
    }
    l[threadIdx.x] = row_m_prev + __logf(row_l_prev);
}

int main() {
    const int S1 = 2048;
    const int S2 = 2048;
    const int dk = 64;
    const int H = 12;
    const int Bc = 32;
    const int Br = 32;
    const int Tc = CEIL_DIV(S2, Bc);
    const int Tr = CEIL_DIV(S1, Br);
    // Q, K, V tile, S tile
    const int sram_size = (Br * dk + 2 * Bc * dk + Br * Bc);

    float *Q = (float *)malloc(H * S1 * dk * sizeof(float));
    float *K = (float *)malloc(H * S2 * dk * sizeof(float));
    float *V = (float *)malloc(H * S2 * dk * sizeof(float));
    float *l = (float *)malloc(H * S1 * sizeof(float));
    float *O_gpu_cal = (float *)malloc(H * S1 * dk * sizeof(float));
    float *O_cpu_cal = (float *)malloc(H * S1 * dk * sizeof(float));

    generateRandomFloatArray(Q, H * S1 * dk);
    generateRandomFloatArray(K, H * S2 * dk);
    generateRandomFloatArray(V, H * S2 * dk);

    float *d_Q, *d_K, *d_V, *d_l, *d_O;
    cudaMalloc((void **)&d_Q, H * S1 * dk * sizeof(float));
    cudaMalloc((void **)&d_K, H * S2 * dk * sizeof(float));
    cudaMalloc((void **)&d_V, H * S2 * dk * sizeof(float));
    cudaMalloc((void **)&d_l, H * S1 * sizeof(float));
    cudaMalloc((void **)&d_O, H * S1 * dk * sizeof(float));

    cudaMemset(d_O, 0, H * S1 * dk * sizeof(float));
    cudaMemset(d_l, 0, H * S1 * sizeof(float));

    cudaMemcpy(d_Q, Q, H * S1 * dk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, H * S2 * dk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, H * S2 * dk * sizeof(float), cudaMemcpyHostToDevice);

    // 每个 block 处理 [Br,dk] 个元素
    dim3 grid(H, Tr);
    dim3 block(Br);
    for(int iter = 0; iter < 5; iter++) {
        Perf("flash_attn_v2");
        flash_attn_v2<sram_size><<<grid, block>>>(d_Q, d_K, d_V, d_l, d_O, S1, dk, Tc, Tr, Bc, Br);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(O_gpu_cal, d_O, H * S1 * dk * sizeof(float), cudaMemcpyDeviceToHost);

    cpu_multihead_attention(Q, K, V, O_cpu_cal, H, S1, S2, dk);
    printFloatArray(O_cpu_cal, 10);
    printFloatArray(O_gpu_cal, 10);
    compare_matrices(H, S1, dk, O_cpu_cal, O_gpu_cal);

    free(Q);
    free(K);
    free(V);
    free(l);
    free(O_gpu_cal);
    free(O_cpu_cal);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
}