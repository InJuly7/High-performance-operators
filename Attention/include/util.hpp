#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>
#include <float.h>


// Q [H, N1, dk] K [H, N2, dk] V[H, N2, dk]
// Attention_out = safe-softmax(\frac{QK^{T}}{d})V
void cpu_multihead_attention(float *Q, float *K, float *V, float *O, int N1, int N2, int dk, int head_num) {
    float rsqrt_dk = 1.0f / sqrtf(dk);
    // safe-softmax 一行的结果
    float *S = (float *)malloc(N2 * sizeof(float));

    // 遍历每个头
    for (int h = 0; h < head_num; ++h) {
        // Q, K, V的头偏移
        int head_offset = h * N1 * dk;
        int out_offset = h * N1 * dk;

        float *Q_h = Q + head_offset;
        float *K_h = K + head_offset;
        float *V_h = V + head_offset;
        float *O_h = O + out_offset;

        for (int i = 0; i < N1; ++i) {
            // 计算 Q[i] @ K.T
            for (int j = 0; j < N2; ++j) {
                float temp = 0.0f;
                for (int k = 0; k < dk; ++k) {
                    temp += Q_h[i * dk + k] * K_h[j * dk + k];
                }
                S[j] = temp * rsqrt_dk;
            }

            // Safe softmax
            float max_val = -FLT_MAX;
            for (int j = 0; j < N2; ++j) {
                max_val = fmaxf(max_val, S[j]);
            }
            float sum_val = 0.0f;
            for (int j = 0; j < N2; ++j) {
                S[j] = expf(S[j] - max_val);
                sum_val += S[j];
            }
            for (int j = 0; j < N2; ++j) {
                S[j] /= sum_val;
            }

            // 计算输出 S @ V
            for (int k = 0; k < dk; ++k) {
                float temp = 0.0f;
                for (int j = 0; j < N2; ++j) {
                    temp += S[j] * V_h[j * dk + k];
                }
                O_h[i * dk + k] = temp;
            }
        }
    }

    free(S);
}


void compare_matrices_3d(int H, int N1, int N2, float* cpu_res, float* gpu_res) {
    float epsilon = 1e-5;
    float tolerance = 1e-3;
    
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N2; ++j) {
                int idx = h * N1 * N2 + i * N1 + j;
                // 计算绝对误差
                float abso_error = fabs(cpu_res[idx] - gpu_res[idx]);
                // 计算相对误差
                float rela_error = abso_error / (fmax(fabs(cpu_res[idx]), fabs(gpu_res[idx])) + epsilon);
                
                if (abso_error > tolerance && rela_error > tolerance) {
                    printf("error: (%d,%d,%d) : cpu_res : %f gpu_res = %f\n", h, i, j, cpu_res[idx], gpu_res[idx]);
                    return;
                }
            }
        }
    }
    printf("cpu_res == gpu_res\n");
}

void generateRandomFloatArray(float* arr, int N) {
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 生成随机数
    for (int i = 0; i < N; i++) {
        arr[i] = dis(gen);
    }
}

void printFloatArray(float* arr, int N) {
    for (int i = 0; i < N; i++) {
        printf("%f ", arr[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

#endif