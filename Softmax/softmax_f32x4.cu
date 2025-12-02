#include "../include/common.hpp"
#include "../include/pybind.hpp"
#include "../include/kernel.cuh"

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ float warpsum[NUM_WARPS];
    val = warp_reduce_sum_f32(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    // tid == 0 返回 block_reduce_sum
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : 0.0f;
        val = warp_reduce_sum_f32(val);
    }
    return val;
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// e^x_i/sum(e^x_0,...,e^x_n-1)
template <unsigned int NUM_THREADS>
__global__ void softmax_f32x4_kernel(float* mat_A, float* mat_B, int N) {
    float* thread_A_start = mat_A + blockIdx.x * N + 4 * threadIdx.x;
    float* thread_B_start = mat_B + blockIdx.x * N + 4 * threadIdx.x;
    float4 reg_A = FLOAT4(thread_A_start[0]);
    reg_A.x = __expf(reg_A.x), reg_A.y = __expf(reg_A.y), reg_A.z = __expf(reg_A.z), reg_A.w = __expf(reg_A.w);
    __shared__ float exp_sum;
    float exp_val = reg_A.x + reg_A.y + reg_A.z + reg_A.w;
    float local_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();

    float4 reg_B;
    reg_B.x = __fdividef(reg_A.x, exp_sum), reg_B.y = __fdividef(reg_A.y, exp_sum), reg_B.z = __fdividef(reg_A.z, exp_sum),
    reg_B.w = __fdividef(reg_A.w, exp_sum);
    FLOAT4(thread_B_start[0]) = reg_B;
}

#define LAUNCH_SOFTMAX_F32x4(H) \
    softmax_f32x4_kernel<((H) / 4)><<<grid, block>>>(reinterpret_cast<float*>(A.data_ptr()), reinterpret_cast<float*>(B.data_ptr()), (H));

#define DISPATCH_SOFTMAX_F32x4(S, H)                                      \
    dim3 block((H / 4));                                                  \
    dim3 grid((S));                                                       \
    switch ((H)) {                                                        \
        case 128:                                                         \
            LAUNCH_SOFTMAX_F32x4(128) break;                              \
        case 256:                                                         \
            LAUNCH_SOFTMAX_F32x4(256) break;                              \
        case 512:                                                         \
            LAUNCH_SOFTMAX_F32x4(512) break;                              \
        case 1024:                                                        \
            LAUNCH_SOFTMAX_F32x4(1024) break;                             \
        case 2048:                                                        \
            LAUNCH_SOFTMAX_F32x4(2048) break;                             \
        case 4096:                                                        \
            LAUNCH_SOFTMAX_F32x4(4096) break;                             \
        default:                                                          \
            throw std::runtime_error("only support H: 128/256/.../4096"); \
            break;                                                        \
    }

void softmax_f32x4(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(A, B)
    const int S = A.size(0);  // seqlens
    const int H = A.size(1);  // head size/kv_len
    // const int N = S * H;
    DISPATCH_SOFTMAX_F32x4(S, H)
}
