#ifndef UTIL_HPP // UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>

class Perf {
   public:
    Perf(const std::string &name) {
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

bool check(float *cpu_result, float *gpu_result, const int M, const int N) {
    const int size = M * N;
    for (int i = 0; i < size; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            return false;
        }
    }
    return true;
}

void transpose_cpu(float *input, float *output, const int M, const int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            const int input_index = m * N + n;
            const int output_index = n * M + m;
            output[output_index] = input[input_index];
        }
    }
}

#endif // UTIL_HPP