#ifndef CUDA_LOG_CUH
#define CUDA_LOG_CUH

#define WARP_SIZE 32

#include <iostream>

// 只支持 grid(1,1,1) block(BLOCK_SIZE, 1 ,1)
// 可变参数模板 : Args 可以接受 0 个或多个类型参数

// 完美转发：&& 是通用引用 ...args 是参数包展开
template <typename... Args>
__device__ __forceinline__ void cudaLog(const char *fmt = "", Args &&...args) {
    // warp_size 32
    const int warp_id = threadIdx.x / WARP_SIZE;
    char full_fmt[50] = "[W(%d)T(%d)]";

#define STRLEN(s)            \
    ({                       \
        const char *p = (s); \
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

#endif