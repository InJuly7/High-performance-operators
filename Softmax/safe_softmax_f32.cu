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

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

template <unsigned int NUM_THREADS>
__device__ __forceinline__ float block_reduce_max_f32(float val) {
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ float warpmax[NUM_WARPS];
    val = warp_reduce_max_f32(val);
    if (laneId == 0) warpmax[warpId] = val;
    __syncthreads();
    // tid == 0 返回 block_reduce_max
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpmax[laneId] : -FLOAT_MAX;
        val = warp_reduce_max_f32(val);
    }
    return val;
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS
// e^x_i/sum(e^x_0,...,e^x_n-1)
template <unsigned int NUM_THREADS>
__global__ void safe_softmax_f32_kernel(float* mat_A, float* mat_B, int N) {
    float* thread_A_start = mat_A + blockIdx.x * N + threadIdx.x;
    float* thread_B_start = mat_B + blockIdx.x * N + threadIdx.x;

    __shared__ float exp_sum;
    __shared__ float global_max;
    float val = thread_A_start[0];
    float local_max = block_reduce_max_f32<NUM_THREADS>(val);
    if (threadIdx.x == 0) global_max = local_max;
    __syncthreads();
    float exp_val = expf(val - global_max);
    float local_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();
    thread_B_start[0] = exp_val / exp_sum;
}

// safe softmax
#define LAUNCH_SAFE_SOFTMAX_F32(H) \
    safe_softmax_f32_kernel<(H)><<<grid, block>>>(reinterpret_cast<float*>(A.data_ptr()), reinterpret_cast<float*>(B.data_ptr()), H);

#define DISPATCH_SAFE_SOFTMAX_F32(S, H)                                 \
    dim3 block((H));                                                    \
    dim3 grid((S));                                                     \
    switch ((H)) {                                                      \
        case 32:                                                        \
            LAUNCH_SAFE_SOFTMAX_F32(32) break;                          \
        case 64:                                                        \
            LAUNCH_SAFE_SOFTMAX_F32(64) break;                          \
        case 128:                                                       \
            LAUNCH_SAFE_SOFTMAX_F32(128) break;                         \
        case 256:                                                       \
            LAUNCH_SAFE_SOFTMAX_F32(256) break;                         \
        case 512:                                                       \
            LAUNCH_SAFE_SOFTMAX_F32(512) break;                         \
        case 1024:                                                      \
            LAUNCH_SAFE_SOFTMAX_F32(1024) break;                        \
        default:                                                        \
            throw std::runtime_error("only support H: 32/64/.../1024"); \
            break;                                                      \
    }

void safe_softmax_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(A, B)
    const int S = A.size(0);  // seqlens
    const int H = B.size(1);  // head size/kv_len
    // const int N = S * H;
    DISPATCH_SAFE_SOFTMAX_F32(S, H)
}