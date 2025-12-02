#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "./common.hpp"

#define WARP_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Vector Access
#define HALF2(value) (reinterpret_cast<half2*>(&(value)))[0]
#define HALF4(value) (reinterpret_cast<float2*>(&(value)))[0]
#define HALF8(value) (reinterpret_cast<float4*>(&(value)))[0]
#define FLOAT2(val) (reinterpret_cast<float2*>(&(value)))[0]
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val)))[0]
#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value)))[0]
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value)))[0]
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value)))[0]
#define REG(val) (*reinterpret_cast<uint32_t*>(&(val)))

// PTX ISA
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                 \
    asm volatile(                                                           \
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, " \
        "[%4];\n"                                                           \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                            \
        : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)             \
    asm volatile(                                                               \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
        : "=r"(RD0), "=r"(RD1)                                                  \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#define FLOAT_MAX FLT_MAX
#define HALF_MAX (__half(65504.0f))

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

// 只支持 grid(1,1,1) block(BLOCK_SIZE, 1 ,1)
// 可变参数模板 : Args 可以接受 0 个或多个类型参数
// 完美转发：&& 是通用引用 ...args 是参数包展开
template <typename... Args>
__device__ __forceinline__ void cudaLog(const char* fmt = "", Args&&... args) {
    // warp_size 32
    const int warp_id = threadIdx.x / WARP_SIZE;
    char full_fmt[50] = "[W(%d)T(%d)]";

#define STRLEN(s)            \
    ({                       \
        const char* p = (s); \
        size_t len = 0;      \
        while (*p++) len++;  \
        len;                 \
    })
    auto prefix_len = STRLEN(full_fmt);
    // 如 GMem[%d] ==> SMem[%d]
    auto fmt_len = STRLEN(fmt);
#undef STRLEN

    for (auto i = 0; i < fmt_len; i++) {
        full_fmt[prefix_len + i] = fmt[i];
    }

    // 判断是否有附加参数
    if (sizeof...(Args) > 0) {
        printf(full_fmt, warp_id, threadIdx.x, std::forward<Args>(args)...);
    } else {
        printf(full_fmt, warp_id, threadIdx.x);
    }
}
