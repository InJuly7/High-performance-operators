#pragma once
#include "include/common.hpp"

// e^x_i/sum(e^x_0,...,e^x_n-1)
void cpu_safe_softmax(float* mat_A, float* mat_B_cpu_calc, const int N1, const int N2) {
    for (int i = 0; i < N1; i++) {
        // 找到每行的最大值，避免数值溢出
        float max_val = mat_A[i * N2];
        for (int j = 1; j < N2; j++) {
            if (mat_A[i * N2 + j] > max_val) {
                max_val = mat_A[i * N2 + j];
            }
        }

        // 计算 exp(x - max) 的和
        float sum_exp = 0.0f;
        for (int j = 0; j < N2; j++) {
            sum_exp += expf(mat_A[i * N2 + j] - max_val);
        }

        // 归一化
        for (int j = 0; j < N2; j++) {
            mat_B_cpu_calc[i * N2 + j] = expf(mat_A[i * N2 + j] - max_val) / sum_exp;
        }
    }
}

void cpu_safe_softmax(half_t* mat_A, half_t* mat_B_cpu_calc, const int N1, const int N2) {
    using namespace half_float;
    for (int i = 0; i < N1; i++) {
        // 找到每行的最大值，避免数值溢出
        half_t max_val = mat_A[i * N2];
        for (int j = 1; j < N2; j++) {
            int idx = i * N2 + j;
            if (mat_A[idx] > max_val) {
                max_val = mat_A[idx];
            }
        }

        // 计算 exp(x - max) 的和
        half_t sum_exp = (half_t)0.0f;
        for (int j = 0; j < N2; j++) {
            sum_exp += expf(mat_A[i * N2 + j] - max_val);
        }

        // 归一化
        for (int j = 0; j < N2; j++) {
            mat_B_cpu_calc[i * N2 + j] = expf(mat_A[i * N2 + j] - max_val) / sum_exp;
        }
    }
}

void cpu_softmax(float* mat_A, float* mat_B_cpu_calc, const int N1, const int N2) {
    for (int i = 0; i < N1; i++) {
        float sum_exp = 0.0f;
        for (int j = 0; j < N2; j++) {
            sum_exp += expf(mat_A[i * N2 + j]);
        }

        // 第三步：归一化
        for (int j = 0; j < N2; j++) {
            mat_B_cpu_calc[i * N2 + j] = expf(mat_A[i * N2 + j]) / sum_exp;
        }
    }
}