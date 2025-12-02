#include "../include/common.hpp"
#include "../include/pybind.hpp"
#include "../include/kernel.cuh"

// FP16
__device__ struct __align__(8) MD {
    half m;
    half d;
};

template <unsigned int NUM_THREADS>
__device__ __forceinline__ MD warp_reduce_md(MD val) {
#pragma unroll
    for (int delta = NUM_THREADS >> 1; delta >= 1; delta >>= 1) {
        MD other;
        other.m = __shfl_down_sync(0xffffffff, val.m, delta);
        other.d = __shfl_down_sync(0xffffffff, val.d, delta);

        MD bigger_MD = val.m >= other.m ? val : other;
        MD smaller_MD = val.m < other.m ? val : other;

        val.d = bigger_MD.d + smaller_MD.d * hexp(smaller_MD.m - bigger_MD.m);
        val.m = bigger_MD.m;
    }
    return val;
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// e^x_i/sum(e^x_0,...,e^x_n-1)
template <unsigned int NUM_THREADS>
__global__ void online_softmax_f16x8_kernel(half* mat_A, half* mat_B, int N) {
    half* thread_A_start = mat_A + blockIdx.x * N + 8 * threadIdx.x;
    half* thread_B_start = mat_B + blockIdx.x * N + 8 * threadIdx.x;
    __shared__ MD block_val;
    half2 reg_A_0 = HALF2(thread_A_start[0]);
    half2 reg_A_1 = HALF2(thread_A_start[2]);
    half2 reg_A_2 = HALF2(thread_A_start[4]);
    half2 reg_A_3 = HALF2(thread_A_start[6]);
    // 默认1.0f, 代表假设每一个元素都是 max, e^{x-m} = 1.0f;
    MD val;
    half temp1 = __hmax(reg_A_0.x, reg_A_0.y);
    half temp2 = __hmax(reg_A_1.x, reg_A_1.y);
    half temp3 = __hmax(reg_A_2.x, reg_A_2.y);
    half temp4 = __hmax(reg_A_3.x, reg_A_3.y);
    half temp5 = __hmax(temp1, temp2);
    half temp6 = __hmax(temp3, temp4);
    val.m = __hmax(temp5, temp6);
    temp1 = hexp(reg_A_0.x - val.m) + hexp(reg_A_0.y - val.m);
    temp2 = hexp(reg_A_1.x - val.m) + hexp(reg_A_1.y - val.m);
    temp3 = hexp(reg_A_2.x - val.m) + hexp(reg_A_2.y - val.m);
    temp4 = hexp(reg_A_3.x - val.m) + hexp(reg_A_3.y - val.m);
    temp5 = temp1 + temp2;
    temp6 = temp3 + temp4;
    val.d = temp5 + temp6;

    const int WARP_NUM = CEIL_DIV(NUM_THREADS, WARP_SIZE);
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ MD warp_MD[WARP_NUM];

    MD warp_val = warp_reduce_md<WARP_SIZE>(val);
    if (laneId == 0) warp_MD[warpId] = warp_val;
    __syncthreads();
    if (warpId == 0) {
        val = (laneId < WARP_NUM) ? warp_MD[laneId] : MD{-HALF_MAX, __float2half(0.0f)};
        val = warp_reduce_md<WARP_NUM>(val);
        if (laneId == 0) block_val = val;
    }
    __syncthreads();

    reg_A_0.x = __hdiv(hexp(reg_A_0.x - block_val.m), block_val.d);
    reg_A_0.y = __hdiv(hexp(reg_A_0.y - block_val.m), block_val.d);
    reg_A_1.x = __hdiv(hexp(reg_A_1.x - block_val.m), block_val.d);
    reg_A_1.y = __hdiv(hexp(reg_A_1.y - block_val.m), block_val.d);
    reg_A_2.x = __hdiv(hexp(reg_A_2.x - block_val.m), block_val.d);
    reg_A_2.y = __hdiv(hexp(reg_A_2.y - block_val.m), block_val.d);
    reg_A_3.x = __hdiv(hexp(reg_A_3.x - block_val.m), block_val.d);
    reg_A_3.y = __hdiv(hexp(reg_A_3.y - block_val.m), block_val.d);

    HALF2(thread_B_start[0]) = reg_A_0;
    HALF2(thread_B_start[2]) = reg_A_1;
    HALF2(thread_B_start[4]) = reg_A_2;
    HALF2(thread_B_start[6]) = reg_A_3;
}

// online safe-softmax
#define LAUNCH_ONLINE_SOFTMAX_F16x8(H) \
    online_softmax_f16x8_kernel<(H) / 8><<<grid, block>>>(reinterpret_cast<half*>(A.data_ptr()), reinterpret_cast<half*>(B.data_ptr()), H);

#define DISPATCH_ONLINE_SOFTMAX_F16x8(S, H)                               \
    dim3 block((H) / 8);                                                  \
    dim3 grid((S));                                                       \
    switch ((H)) {                                                        \
        case 256:                                                         \
            LAUNCH_ONLINE_SOFTMAX_F16x8(256) break;                       \
        case 512:                                                         \
            LAUNCH_ONLINE_SOFTMAX_F16x8(512) break;                       \
        case 1024:                                                        \
            LAUNCH_ONLINE_SOFTMAX_F16x8(1024) break;                      \
        case 2048:                                                        \
            LAUNCH_ONLINE_SOFTMAX_F16x8(2048) break;                      \
        case 4096:                                                        \
            LAUNCH_ONLINE_SOFTMAX_F16x8(4096) break;                      \
        case 8192:                                                        \
            LAUNCH_ONLINE_SOFTMAX_F16x8(8192) break;                      \
        default:                                                          \
            throw std::runtime_error("only support H: 256/512/.../8192"); \
            break;                                                        \
    }

void online_softmax_f16x8(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat16)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat16)
    CHECK_TORCH_TENSOR_SHAPE(A, B)
    const int S = A.size(0);  // seqlens
    const int H = B.size(1);  // head size/kv_len
    // const int N = S * H;
    DISPATCH_ONLINE_SOFTMAX_F16x8(S, H)
}