#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <random>

void cpu_sgemm(float* mat_A, float* mat_B, float* mat_C_cpu_cal, const int M, const int K, const int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += mat_A[m * K + k] * mat_B[k * N + n];
            }
            mat_C_cpu_cal[m * N + n] = temp;
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

void bank_conflict_label(std::string str, int tid, int row[32], int col[32], int SMem_Row, int SMem_Col, int OFFSET) {
    int count[32] = {0};
    int warpId = tid / 32;

    printf("\n=== %s Bank Conflict Analysis ===\n", str.c_str());
    printf("WarpId | TID | Row | Col | Logic Addr | Phys Addr | BankId\n");
    printf("-------|-----|-----|-----|------------|-----------|-------\n");

    for (int i = tid; i < tid + 32; i++) {
        int VAddr = row[i] * SMem_Col + col[i];
        int PAddr = row[i] * (SMem_Col + OFFSET) + col[i];
        int bankId = PAddr % 32;
        count[bankId]++;

        printf(" %5d | %3d | %3d | %3d | %10d | %9d | %5d \n", warpId, i, row[i], col[i], VAddr, PAddr, bankId);
    }

    // 计算最大bank conflict
    int bfc = 1;
    for (int i = 0; i < 32; i++) {
        bfc = std::max(bfc, count[i]);
    }

    printf("\n=== Bank Usage ===\n");
    for (int i = 0; i < 32; i++) {
        if (count[i] > 1) {
            printf("Bank %2d: %d accesses (CONFLICT!)\n", i, count[i]);
        } else if (count[i] == 1) {
            printf("Bank %2d: %d access\n", i, count[i]);
        }
    }
    if(bfc > 1) printf("\nResult: %d-way bank conflict\n\n", bfc);
    else printf("\nNo Bank conflict\n\n");
}

#endif