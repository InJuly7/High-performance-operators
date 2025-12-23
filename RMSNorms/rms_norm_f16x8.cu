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
    const int NUM_WARPS = CEIL_DIV(NUM_THREADS, WARP_SIZE);
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ float warpsum[NUM_WARPS];
    val = warp_reduce_sum_f32(val);
    if (laneId == 0) warpsum[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (laneId < NUM_WARPS) ? warpsum[laneId] : 0.0f;
        val = warp_reduce_sum_f32(val);
    }
    return val;
}

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)

template <unsigned int NUM_THREADS>
__global__ void rms_norm_f16x8_kernel(half* A, half* B, half* C, int K) {
    half* thread_A_start = A + blockIdx.x * K + threadIdx.x * 8;
    half* thread_B_start = B + threadIdx.x * 8; 
    half* thread_C_start = C + blockIdx.x * K + threadIdx.x * 8;
    const float epsilon = 1e-5f;
    // 块内共享, 求出当前行 rsqrtf(sum(ai^2)/K)
    __shared__ float s_variance;

    float2 reg_A_0 = __half22float2(HALF2(thread_A_start[0]));
    float2 reg_A_1 = __half22float2(HALF2(thread_A_start[2]));
    float2 reg_A_2 = __half22float2(HALF2(thread_A_start[4]));
    float2 reg_A_3 = __half22float2(HALF2(thread_A_start[6]));

    float2 reg_B_0 = __half22float2(HALF2(thread_B_start[0]));
    float2 reg_B_1 = __half22float2(HALF2(thread_B_start[2]));
    float2 reg_B_2 = __half22float2(HALF2(thread_B_start[4]));
    float2 reg_B_3 = __half22float2(HALF2(thread_B_start[6]));
    
    float variance = reg_A_0.x * reg_A_0.x + reg_A_0.y * reg_A_0.y;
    variance += reg_A_1.x * reg_A_1.x + reg_A_1.y * reg_A_1.y;
    variance += reg_A_2.x * reg_A_2.x + reg_A_2.y * reg_A_2.y;
    variance += reg_A_3.x * reg_A_3.x + reg_A_3.y * reg_A_3.y;

    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) s_variance = rsqrtf(variance / K + epsilon);
    __syncthreads();

    reg_A_0.x = reg_A_0.x * reg_B_0.x * s_variance;
    reg_A_0.y = reg_A_0.y * reg_B_0.y * s_variance;
    reg_A_1.x = reg_A_1.x * reg_B_1.x * s_variance;
    reg_A_1.y = reg_A_1.y * reg_B_1.y * s_variance;
    reg_A_2.x = reg_A_2.x * reg_B_2.x * s_variance;
    reg_A_2.y = reg_A_2.y * reg_B_2.y * s_variance;
    reg_A_3.x = reg_A_3.x * reg_B_3.x * s_variance;
    reg_A_3.y = reg_A_3.y * reg_B_3.y * s_variance;

    HALF2(thread_C_start[0]) = __float22half2_rn(reg_A_0);
    HALF2(thread_C_start[2]) = __float22half2_rn(reg_A_1);
    HALF2(thread_C_start[4]) = __float22half2_rn(reg_A_2);
    HALF2(thread_C_start[6]) = __float22half2_rn(reg_A_3);
}

#define LAUNCH_RMS_NORM_F16x8(H)                                                                                                  \
    rms_norm_f16x8_kernel<(H) / 8><<<grid, block>>>(reinterpret_cast<half*>(A.data_ptr()), reinterpret_cast<half*>(B.data_ptr()), \
                                                    reinterpret_cast<half*>(C.data_ptr()), (H));

#define DISPATCH_RMS_NORM_F16x8(S, H)                                  \
    dim3 block((H) / 8);                                                  \
    dim3 grid((S));                                                       \
    switch ((H)) {                                                        \
        case 256:                                                         \
            LAUNCH_RMS_NORM_F16x8(256) break;                          \
        case 512:                                                         \
            LAUNCH_RMS_NORM_F16x8(512) break;                             \
        case 1024:                                                        \
            LAUNCH_RMS_NORM_F16x8(1024) break;                            \
        case 2048:                                                        \
            LAUNCH_RMS_NORM_F16x8(2048) break;                         \
        case 4096:                                                        \
            LAUNCH_RMS_NORM_F16x8(4096) break;                         \
        case 8192:                                                        \
            LAUNCH_RMS_NORM_F16x8(8192) break;                         \
        default:                                                          \
            throw std::runtime_error("only support H: 256/512/.../8192"); \
            break;                                                        \
    }

void rms_norm_f16x8(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat16)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat16)
    CHECK_TORCH_TENSOR_DTYPE(C, torch::kFloat16)
    const int S = A.size(0);  // seqlens
    const int H = A.size(1);
    // const int N = S * H;
    DISPATCH_RMS_NORM_F16x8(S, H)
}
