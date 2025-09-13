#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>
#include <math.h>
#include "/home/song/program/High-performance-operators/include/half.hpp"


using half_t = half_float::half;

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
    using namespace half_float;
    for (int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

void compare_matrices(int N, int K, float* cpu_res, float* gpu_res) {
    int flag = 0;
    float epsilon = 1e-5;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            int idx = i * K + j;
            if (abs(cpu_res[idx] - gpu_res[idx])/(cpu_res[idx] + epsilon) > 1e-3f) {
                printf("error: (%d,%d) : cpu_res : %f gpu_res = %f\n", i, j, cpu_res[i * K + j], gpu_res[i * K + j]);
                flag = 1;
                return;
            }
        }
    }
    if (flag == 0) {
        printf("cpu_res == gpu_res\n");
    }
}

void compare_matrices(int N, int K, half_t* cpu_res, half_t* gpu_res) {
    // using namespace half_float;
    int flag = 0;
    // 适合half精度的epsilon
    half_t eps = (half_t)1e-4f;
    half_t tolerance(1e-2f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            int idx = i * K + j;
            // 计算相对误差
            half_t relative_error = fabs(cpu_res[idx] - gpu_res[idx]);
            if (relative_error > tolerance) {
                std::cout << "error: (" << i << "," << j << ") : cpu_res : " << cpu_res[idx] << "," << gpu_res[idx] << std::endl;
                flag = 1;
                return;
            }
        }
    }

    if (flag == 0) {
        printf("cpu_res == gpu_res \n");
    }
}

void cpu_rms_norm(float* mat_A, float* mat_B_cpu_calc, const float g, const int N, const int K) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < N; i++) {
        float val = 0.0f;  
        for (int j = 0; j < K; j++) {
            val += mat_A[i * K + j] * mat_A[i * K + j];
        }
        val = rsqrtf(val / (float)K + epsilon);
        for (int j = 0; j < K; j++) {
            mat_B_cpu_calc[i * K + j] = (mat_A[i * K + j] * val) * g;
        }
    }
}

void cpu_rms_norm(half_t* mat_A, half_t* mat_B_cpu_calc, const float g, const int N, const int K) {
    using namespace half_float;
    const half_t epsilon = half_t(1e-5f);
    const half_t K_ = half_cast<half_t, int>(K);
    const half_t g_ = half_cast<half_t, float>(g);

    for (int i = 0; i < N; i++) {
        half_t val = half_t(0.0f);

        // 计算平方和
        for (int j = 0; j < K; j++) {
            val += mat_A[i * K + j] * mat_A[i * K + j];
        }

        // 计算 RMS normalization 因子
        val = rsqrt(val / K_ + epsilon);

        // 应用 normalization 和缩放
        for (int j = 0; j < K; j++) {
            mat_B_cpu_calc[i * K + j] = mat_A[i * K + j] * val * g;
        }
    }
}

#endif // UTIL_HPP