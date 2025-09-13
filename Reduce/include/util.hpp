#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>

float cpu_reduce(float* h_a, int N) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        result += h_a[i];
    }
    return result;
}

void compare_matrices(float cpu_result, float gpu_result) {
    if (abs(gpu_result - cpu_result) / cpu_result > 1e-3)
        printf("Gpu Result Error!!\n");
    else
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

#endif