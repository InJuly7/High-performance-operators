#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>
#include <vector>

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

void compare_matrices(int N, float* cpu_res, float* gpu_res) {
    int flag = 0;
    for (int i = 0; i < N; i++) {
        if (abs(cpu_res[i] - gpu_res[i]) > 0.5f) {
            printf("error: %d : cpu_res : %f gpu_res = %f\n", i, cpu_res[i], gpu_res[i]);
            flag = 1;
            break;
        }
    }
    if (flag == 0) {
        printf("cpu_res == gpu_res\n");
    }
}

std::vector<float> generateRandomVector(int N) {
    std::vector<float> vec(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

#endif