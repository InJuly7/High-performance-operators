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
__global__ void rms_norm_f32_kernel(float* A, float* B, float* C, int K) {
    float* A_start = A + blockIdx.x * K;
    float* C_start = C + blockIdx.x * K;
    const float epsilon = 1e-5f;

    // 块内共享, 求出当前行 rsqrtf(sum(ai^2)/K)
    __shared__ float s_variance;
    float value = A_start[threadIdx.x];
    float variance = value * value;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();
    C_start[threadIdx.x] = (value * s_variance) * B[threadIdx.x];
}

#define LAUNCH_RMS_NORM_F32(H)                                                                                                \
    rms_norm_f32_kernel<(H)><<<grid, block>>>(reinterpret_cast<float*>(A.data_ptr()), reinterpret_cast<float*>(B.data_ptr()), \
                                              reinterpret_cast<float*>(C.data_ptr()), (H));

#define DISPATCH_RMS_NORM_F32(S, H)                                     \
    dim3 block((H));                                                    \
    dim3 grid((S));                                                     \
    switch ((H)) {                                                      \
        case 32:                                                        \
            LAUNCH_RMS_NORM_F32(32) break;                              \
        case 64:                                                        \
            LAUNCH_RMS_NORM_F32(64) break;                              \
        case 128:                                                       \
            LAUNCH_RMS_NORM_F32(128) break;                             \
        case 256:                                                       \
            LAUNCH_RMS_NORM_F32(256) break;                             \
        case 512:                                                       \
            LAUNCH_RMS_NORM_F32(512) break;                             \
        case 1024:                                                      \
            LAUNCH_RMS_NORM_F32(1024) break;                            \
        default:                                                        \
            throw std::runtime_error("only support H: 32/64/.../1024"); \
            break;                                                      \
    }

void rms_norm_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(C, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(A, C)
    const int S = A.size(0);  // seqlens
    const int H = A.size(1);
    // const int N = S * H;
    DISPATCH_RMS_NORM_F32(S, H)
}