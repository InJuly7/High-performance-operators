#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>
#include <vector>

void cpu_sgemm(float* matrix_A_host, float* matrix_B_host, float* matrix_C_host_cpu_calc, const int M, const int N, const int K) {
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