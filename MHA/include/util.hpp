#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>
#include <float.h>


class Perf
{
public:
    Perf(const std::string &name)
    {
        m_name = name;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }

    ~Perf()
    {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0.0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        std::cout << m_name << " elapse: " << elapsed_time << " ms" << std::endl;
    }

private:
    std::string m_name;
    cudaEvent_t m_start, m_end;
}; // class Perf


void cpu_qk_matmul(float *Q, float *K, float *S, int H, int S1, int S2, int dk) {
    float scale = rsqrtf(dk);
    // 遍历每个头
    for (int h = 0; h < H; ++h) {
        // Q, K, S的头偏移
        int Q_head_offset = h * S1 * dk;
        int K_head_offset = h * S2 * dk;
        int S_head_offset = h * S1 * S2;

        float *Q_h = Q + Q_head_offset;
        float *K_h = K + K_head_offset;
        float *S_h = S + S_head_offset;

        for (int i = 0; i < S1; ++i) {
            // 计算 Q[i] @ K.T
            for (int j = 0; j < S2; ++j) {
                float temp = 0.0f;
                for (int k = 0; k < dk; ++k) {
                    temp += Q_h[i * dk + k] * K_h[j * dk + k];
                }
                S_h[i * S2 + j] = temp * scale;
            }
        }
    }
}

void cpu_safe_softmax(float *S, float *P, const int H, const int S1, const int S2) {
    for (int h = 0; h < H; ++h) {
        // S, P的头偏移
        int P_head_offset = h * S1 * S2;
        int S_head_offset = h * S1 * S2;

        float *S_h = S + S_head_offset;
        float *P_h = P + P_head_offset;

        for (int i = 0; i < S1; i++) {
            // 找到每行的最大值，避免数值溢出
            float max_val = S_h[i * S2];
            for (int j = 1; j < S2; j++) {
                if (S_h[i * S2 + j] > max_val) {
                    max_val = S_h[i * S2 + j];
                }
            }

            // 计算 exp(x - max) 的和
            float sum_exp = 0.0f;
            for (int j = 0; j < S2; j++) {
                sum_exp += expf(S_h[i * S2 + j] - max_val);
            }

            // 归一化
            for (int j = 0; j < S2; j++) {
                P_h[i * S2 + j] = expf(S_h[i * S2 + j] - max_val) / sum_exp;
            }
        }
    }
}

void cpu_pv_matmul(float *P, float *V, float *O, int H, int S1, int S2, int dk) {
    // 遍历每个头
    for (int h = 0; h < H; ++h) {
        // Q, K, S的头偏移
        int P_head_offset = h * S1 * S2;
        int V_head_offset = h * S2 * dk;
        int O_head_offset = h * S1 * dk;

        float *P_h = P + P_head_offset;
        float *V_h = V + V_head_offset;
        float *O_h = O + O_head_offset;

        for (int i = 0; i < S1; ++i) {
            // 计算 O + P @ V
            for (int j = 0; j < dk; ++j) {
                float temp = 0.0f;
                for (int k = 0; k < S2; ++k) {
                    temp += P_h[i * S2 + k] * V_h[k * dk + j];
                }
                O_h[i * dk + j] = temp;
            }
        }
    }
}

// Q [H, S1, dk] K [H, S2, dk] V[H, S2, dk]
// Attention_out = safe-softmax(\frac{QK^{T}}{d})V
void cpu_multihead_attention(float *Q, float *K, float *V, float *O, int H, int S1, int S2, int dk) {
    float rsqrt_dk = rsqrtf(dk);
    // safe-softmax 一行的结果
    float *S = (float *)malloc(S2 * sizeof(float));

    // 遍历每个头
    for (int h = 0; h < H; ++h) {
        // Q, K, V的头偏移
        int Q_head_offset = h * S1 * dk;
        int K_head_offset = h * S2 * dk;
        int O_head_offset = h * S1 * dk;

        float *Q_h = Q + Q_head_offset;
        float *K_h = K + K_head_offset;
        float *V_h = V + K_head_offset;
        float *O_h = O + O_head_offset;

        for (int i = 0; i < S1; ++i) {
            // 计算 Q[i] @ K.T
            // attn-mask
            for (int j = 0; j < S2; ++j) {
                float temp = 0.0f;
                for (int k = 0; k < dk; ++k) {
                    temp += Q_h[i * dk + k] * K_h[j * dk + k];
                }
                temp *= rsqrt_dk;
                S[j] = (j <= i) ? temp : -INFINITY;
            }

            // Safe softmax
            float max_val = -FLT_MAX;
            for (int j = 0; j < S2; ++j) {
                max_val = fmaxf(max_val, S[j]);
            }
            float sum_val = 0.0f;
            for (int j = 0; j < S2; ++j) {
                S[j] = expf(S[j] - max_val);
                sum_val += S[j];
            }
            for (int j = 0; j < S2; ++j) {
                S[j] /= sum_val;
            }

            // 计算输出 S @ V
            for (int k = 0; k < dk; ++k) {
                float temp = 0.0f;
                for (int j = 0; j < S2; ++j) {
                    temp += S[j] * V_h[j * dk + k];
                }
                O_h[i * dk + k] = temp;
            }
        }
    }

    free(S);
}

void compare_matrices(int D1, int D2, int D3, float *cpu_res, float *gpu_res) {
    float epsilon = 1e-5;
    float tolerance = 1e-3;

    for (int h = 0; h < D1; ++h) {
        for (int i = 0; i < D2; ++i) {
            for (int j = 0; j < D3; ++j) {
                int idx = h * D2 * D3 + i * D3 + j;
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

void generateRandomFloatArray(float *arr, int N) {
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 生成随机数
    for (int i = 0; i < N; i++) {
        arr[i] = dis(gen);
    }
}

void printFloatArray(float *arr, int N) {
    for (int i = 0; i < N; i++) {
        printf("%f ", arr[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

#endif