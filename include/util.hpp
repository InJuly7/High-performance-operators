#pragma once
#include "common.hpp"
#include "half.hpp"

using half_t = half_float::half;

void generateRandomHalfArray(half_t* arr, int N) {
    using namespace half_float;
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 生成随机数
    for (int i = 0; i < N; i++) {
        //
        arr[i] = half_t(dis(gen));
    }
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

void printHalfArray(half_t* arr, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

void compare_matrices(int N1, int N2, float* cpu_res, float* gpu_res) {
    float epsilon = 1e-5;
    float tolerance = 1e-3;
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            int idx = i * N2 + j;
            // 计算绝对误差
            float abso_error = fabs(cpu_res[idx] - gpu_res[idx]);
            // 计算相对误差
            float rela_error = abso_error / (fmax(fabs(cpu_res[idx]), fabs(gpu_res[idx])) + epsilon);
            if (abso_error > tolerance && rela_error > tolerance) {
                printf("error: (%d,%d) : cpu_res : %f gpu_res = %f\n", i, j, cpu_res[idx], gpu_res[idx]);
                return;
            }
        }
    }
    printf("cpu_res == gpu_res\n");
}

void compare_matrices(int N1, int N2, half_t* cpu_res, half_t* gpu_res) {
    // 适合half精度的epsilon
    half_t epsilon = (half_t)1e-4f;
    half_t tolerance(1e-2f);
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            int idx = i * N2 + j;
            // 计算绝对误差
            half_t abso_error = fabs(cpu_res[idx] - gpu_res[idx]);
            // 计算相对误差
            half_t rela_error = abso_error / (fmax(fabs(cpu_res[idx]), fabs(gpu_res[idx])) + epsilon);
            if (abso_error > tolerance && rela_error > tolerance) {
                std::cout << "error: (" << i << "," << j << ") : cpu_res : " << cpu_res[idx] << " " << gpu_res[idx] << std::endl;
                return;
            }
        }
    }

    printf("cpu_res == gpu_res \n");
}