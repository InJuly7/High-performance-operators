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
template <unsigned int NUM_THREADS>
__global__ void safe_softmax_f16_pack_kernel(half* mat_A, half* mat_B, int N) {
    half* thread_A_start = mat_A + blockIdx.x * N + 8 * threadIdx.x;
    half* thread_B_start = mat_B + blockIdx.x * N + 8 * threadIdx.x;
    half pack_A[8], pack_B[8];
    LDST128BITS(pack_A[0]) = LDST128BITS(thread_A_start[0]);
    __shared__ half exp_sum;
    __shared__ half global_max;

    half local_max;
    for (int i = 0; i < 8; i++) {
        local_max = __hmax(local_max, pack_A[i]);
    }
    local_max = block_reduce_max_f16<NUM_THREADS>(local_max);
    if (threadIdx.x == 0) global_max = local_max;
    __syncthreads();

    half local_sum;
    for (int i = 0; i < 8; i++) {
        pack_A[i] = hexp(pack_A[i] - global_max);
        local_sum += pack_A[i];
    }
    local_sum = block_reduce_sum_f16<NUM_THREADS>(local_sum);
    if (threadIdx.x == 0) exp_sum = local_sum;
    __syncthreads();

    for (int i = 0; i < 8; i++) {
        pack_B[i] = pack_A[i] / exp_sum;
    }
    LDST128BITS(thread_B_start[0]) = LDST128BITS(pack_B[0]);
}

// safe softmax
#define LAUNCH_SAFE_SOFTMAX_F16_PACK(H) \
    safe_softmax_f16_pack_kernel<(H) / 8><<<grid, block>>>(reinterpret_cast<half*>(A.data_ptr()), reinterpret_cast<half*>(B.data_ptr()), H);

#define DISPATCH_SAFE_SOFTMAX_F16_PACK(S, H)                              \
    dim3 block((H) / 8);                                                  \
    dim3 grid((S));                                                       \
    switch ((H)) {                                                        \
        case 256:                                                         \
            LAUNCH_SAFE_SOFTMAX_F16_PACK(256) break;                      \
        case 512:                                                         \
            LAUNCH_SAFE_SOFTMAX_F16_PACK(512) break;                      \
        case 1024:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16_PACK(1024) break;                     \
        case 2048:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16_PACK(2048) break;                     \
        case 4096:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16_PACK(4096) break;                     \
        case 8192:                                                        \
            LAUNCH_SAFE_SOFTMAX_F16_PACK(8192) break;                     \
        default:                                                          \
            throw std::runtime_error("only support H: 256/512/.../8192"); \
            break;                                                        \
    }

void safe_softmax_f16_pack(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat16)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat16)
    CHECK_TORCH_TENSOR_SHAPE(A, B)
    const int S = A.size(0);  // seqlens
    const int H = B.size(1);  // head size/kv_len
    // const int N = S * H;
    DISPATCH_SAFE_SOFTMAX_F16_PACK(S, H)
}