#include "../include/common.hpp"
#include "../include/pybind.hpp"
#include "../include/kernel.cuh"

__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ half block_reduce_sum_f16(half val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ half warpsum[NUM_WARPS];
    val = warp_reduce_sum_f16(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    // tid == 0 返回 block_reduce_sum
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : __float2half(0.0f);
        val = warp_reduce_sum_f16(val);
    }
    return val;
}

__device__ __forceinline__ half warp_reduce_max_f16(half val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val = __hmax(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ half block_reduce_max_f16(half val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ half warpmax[NUM_WARPS];
    val = warp_reduce_max_f16(val);
    if (laneId == 0) warpmax[warpId] = val;
    __syncthreads();
    // tid == 0 返回 block_reduce_max
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpmax[laneId] : -HALF_MAX;
        val = warp_reduce_max_f16(val);
    }
    return val;
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// e^x_i/sum(e^x_0,...,e^x_n-1)
#define HALF2MAX(reg_x, reg_y) __hmax((reg_x), (reg_y))
#define HALF4MAX(reg_x, reg_y, reg_z, reg_w) __hmax(HALF2MAX(reg_x, reg_y), HALF2MAX(reg_z, reg_w))
#define HALF2_EXP(reg, global_max, local_sum) \
    (reg).x = hexp((reg).x - global_max);     \
    (reg).y = hexp((reg).y - global_max);     \
    local_sum += (reg).x;                     \
    local_sum += (reg).y;
#define HALF2_SOFTMAX(reg_B, reg_A, global_sum) \
    (reg_B).x = __hdiv((reg_A).x, global_sum);  \
    (reg_B).y = __hdiv((reg_A).y, global_sum);

template <unsigned int NUM_THREADS>
__global__ void safe_softmax_f16x8_kernel(half* mat_A, half* mat_B, int N) {
    half* thread_A_start = mat_A + blockIdx.x * N + 8 * threadIdx.x;
    half* thread_B_start = mat_B + blockIdx.x * N + 8 * threadIdx.x;

    __shared__ half exp_sum;
    __shared__ half global_max;

    half local_max;
    half2 reg_A_0 = HALF2(thread_A_start[0]);
    half2 reg_A_1 = HALF2(thread_A_start[2]);
    half2 reg_A_2 = HALF2(thread_A_start[4]);
    half2 reg_A_3 = HALF2(thread_A_start[6]);

    local_max = HALF2MAX(HALF4MAX(reg_A_0.x, reg_A_0.y, reg_A_1.x, reg_A_1.y), HALF4MAX(reg_A_2.x, reg_A_2.y, reg_A_3.x, reg_A_3.y));
    local_max = block_reduce_max_f16<NUM_THREADS>(local_max);
    if (threadIdx.x == 0) global_max = local_max;
    __syncthreads();

    half local_sum;
    HALF2_EXP(reg_A_0, global_max, local_sum);
    HALF2_EXP(reg_A_1, global_max, local_sum);
    HALF2_EXP(reg_A_2, global_max, local_sum);
    HALF2_EXP(reg_A_3, global_max, local_sum);
    local_sum = block_reduce_sum_f16<NUM_THREADS>(local_sum);
    if (threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();

    half2 reg_B_0, reg_B_1, reg_B_2, reg_B_3;
    HALF2_SOFTMAX(reg_B_0, reg_A_0, exp_sum);
    HALF2_SOFTMAX(reg_B_1, reg_A_1, exp_sum);
    HALF2_SOFTMAX(reg_B_2, reg_A_2, exp_sum);
    HALF2_SOFTMAX(reg_B_3, reg_A_3, exp_sum);
    HALF2(thread_B_start[0]) = reg_B_0;
    HALF2(thread_B_start[2]) = reg_B_1;
    HALF2(thread_B_start[4]) = reg_B_2;
    HALF2(thread_B_start[6]) = reg_B_3;
}

// safe softmax
#define LAUNCH_SAFE_SOFTMAX_F16x8(H) \
    safe_softmax_f16x8_kernel<(H) / 8><<<grid, block>>>(reinterpret_cast<half*>(A.data_ptr()), reinterpret_cast<half*>(B.data_ptr()), H);

#define DISPATCH_SAFE_SOFTMAX_F16x8(S, H)                                 \
    dim3 block((H) / 8);                                                  \
    dim3 grid((S));                                                       \
    switch ((H)) {                                                        \
        case 256:                                                         \
            LAUNCH_SAFE_SOFTMAX_F16x8(256) break;                         \
        case 512:                                                         \
            LAUNCH_SAFE_SOFTMAX_F16x8(512) break;                         \
        case 1024:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16x8(1024) break;                        \
        case 2048:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16x8(2048) break;                        \
        case 4096:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16x8(4096) break;                        \
        case 8192:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16x8(8192) break;                        \
        default:                                                          \
            throw std::runtime_error("only support H: 256/512/.../8192"); \
            break;                                                        \
    }

void safe_softmax_f16x8(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat16)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat16)
    CHECK_TORCH_TENSOR_SHAPE(A, B)
    const int S = A.size(0);  // seqlens
    const int H = B.size(1);  // head size/kv_len
    // const int N = S * H;
    DISPATCH_SAFE_SOFTMAX_F16x8(S, H)
}