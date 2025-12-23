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
__global__ void rms_norm_f16_pack_kernel(half* A, half* B, half* C, int K) {
    half* thread_A_start = A + blockIdx.x * K + threadIdx.x * 8;
    half* thread_B_start = B + threadIdx.x * 8;
    half* thread_C_start = C + blockIdx.x * K + threadIdx.x * 8;
    const float epsilon = 1e-5f;
    // 块内共享, 求出当前行 rsqrtf(sum(ai^2)/K)
    __shared__ float s_variance;
    float variance = 0.0f;
    half pack_A[8], pack_B[8];
    float pack_A_f32[8], pack_B_f32[8];
    LDST128BITS(pack_A[0]) = LDST128BITS(thread_A_start[0]);
    LDST128BITS(pack_B[0]) = LDST128BITS(thread_B_start[0]);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_A_f32[i] = __half2float(pack_A[i]);
        pack_B_f32[i] = __half2float(pack_B[i]);
        variance += pack_A_f32[i] * pack_A_f32[i];
    }

    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) s_variance = rsqrtf(variance / K + epsilon);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_A[i] = __float2half(pack_A_f32[i] * s_variance * pack_B_f32[i]);
    }
    LDST128BITS(thread_C_start[0]) = LDST128BITS(pack_A[0]);
}

#define LAUNCH_RMS_NORM_F16_PACK(H)   \
    rms_norm_f16_pack_kernel<(H) / 8> \
        <<<grid, block>>>(reinterpret_cast<half*>(A.data_ptr()), reinterpret_cast<half*>(B.data_ptr()), reinterpret_cast<half*>(C.data_ptr()), (H));

#define DISPATCH_RMS_NORM_F16_PACK(S, H)                                  \
    dim3 block((H) / 8);                                                  \
    dim3 grid((S));                                                       \
    switch ((H)) {                                                        \
        case 256:                                                         \
            LAUNCH_RMS_NORM_F16_PACK(256) break;                          \
        case 512:                                                         \
            LAUNCH_RMS_NORM_F16_PACK(512) break;                          \
        case 1024:                                                        \
            LAUNCH_RMS_NORM_F16_PACK(1024) break;                         \
        case 2048:                                                        \
            LAUNCH_RMS_NORM_F16_PACK(2048) break;                         \
        case 4096:                                                        \
            LAUNCH_RMS_NORM_F16_PACK(4096) break;                         \
        case 8192:                                                        \
            LAUNCH_RMS_NORM_F16_PACK(8192) break;                         \
        default:                                                          \
            throw std::runtime_error("only support H: 256/512/.../8192"); \
            break;                                                        \
    }

void rms_norm_f16_pack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat16)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat16)
    CHECK_TORCH_TENSOR_DTYPE(C, torch::kFloat16)
    const int S = A.size(0);  // seqlens
    const int H = A.size(1);
    // const int N = S * H;
    DISPATCH_RMS_NORM_F16_PACK(S, H)
}
