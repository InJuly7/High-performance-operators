#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>
#include <fstream>
#include "../../include/half.hpp"

using half_t = half_float::half;

class Perf {
   public:
    Perf(const std::string& name) {
        m_name = name;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }

    ~Perf() {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0.0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        std::cout << m_name << " elapse: " << elapsed_time << " ms" << std::endl;
    }

   private:
    std::string m_name;
    cudaEvent_t m_start, m_end;
};  // class Perf

void generateRandomHalfArray(half_t* arr, int N, bool is_write = false, const char* filename = "") {
    using namespace half_float;
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 生成随机数
    for (int i = 0; i < N; i++) {
        arr[i] = half_float::half_cast<half_t, float>(dis(gen));
    }

    if (is_write) {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cout << "filename is empty" << std::endl;
            exit(1);
        }
        for (int i = 0; i < N; i++) {
            out << arr[i] << " ";
            if ((i + 1) % 16 == 0) out << std::endl;
        }
        out.close();
    }
}

void compare_matrices(int N1, int N2, half_t* cpu_res, half_t* gpu_res) {
    // 适合half精度的epsilon
    half_t epsilon = (half_t)1e-3f;
    half_t tolerance(5e-3f);
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            int idx = i * N2 + j;
            // 计算绝对误差
            half_t abso_error = half_float::fabs(cpu_res[idx] - gpu_res[idx]);
            // 计算相对误差
            half_t rela_error = abso_error / (half_float::fmax(half_float::fabs(cpu_res[idx]), half_float::fabs(gpu_res[idx])) + epsilon);
            if (abso_error > tolerance && rela_error > tolerance) {
                std::cout << "abso_error: " << abso_error << " rela_error: " << rela_error << std::endl;
                std::cout << "error: (" << i << "," << j << ") : " << cpu_res[idx] << " " << gpu_res[idx] << std::endl;
                return;
            }
        }
    }

    printf("cpu_res == gpu_res \n");
}

void cpu_hgemm(half_t* mat_A, half_t* mat_B, half_t* mat_C_cpu_cal, const int M, const int K, const int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            half_t temp = half_t(0.0f);
            for (int k = 0; k < K; k++) {
                temp += mat_A[m * K + k] * mat_B[k * N + n];
            }
            mat_C_cpu_cal[m * N + n] = temp;
        }
    }
}

void printHalfArray(half_t* arr, int N) {
    using namespace half_float;
    for (int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

#endif