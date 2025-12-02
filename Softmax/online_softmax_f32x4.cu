#include "../include/common.hpp"
#include "../include/pybind.hpp"
#include "../include/kernel.cuh"

// FP32
// DS required for Online Softmax
__device__ struct __align__(8) MD {
    float m;
    float d;
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
        val.d = bigger_MD.d + smaller_MD.d * __expf(smaller_MD.m - bigger_MD.m);
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
__global__ void online_softmax_f32x4_kernel(float* mat_A, float* mat_B, int N) {
    float* thread_A_start = mat_A + blockIdx.x * N + 4 * threadIdx.x;
    float* thread_B_start = mat_B + blockIdx.x * N + 4 * threadIdx.x;
    const int WARP_NUM = CEIL_DIV(NUM_THREADS, WARP_SIZE);
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ MD warp_MD[WARP_NUM];
    float4 reg_A = FLOAT4(thread_A_start[0]);

    float local_m = fmaxf(fmaxf((reg_A).x, (reg_A).y), fmaxf((reg_A).z, (reg_A).w));
    float local_d = __expf(reg_A.x - local_m) + __expf(reg_A.y - local_m) + __expf(reg_A.z - local_m) + __expf(reg_A.w - local_m);
    MD local_md = {local_m, local_d};
    MD warp_md = warp_reduce_md<WARP_SIZE>(local_md);
    if (laneId == 0) warp_MD[warpId] = warp_md;
    __syncthreads();

    __shared__ MD block_md;
    MD zero_md = {-FLT_MAX, 0.0f};
    if (warpId == 0) {
        local_md = (laneId < WARP_NUM) ? warp_MD[laneId] : zero_md;
        warp_md = warp_reduce_md<WARP_NUM>(local_md);
        if (laneId == 0) block_md = warp_md;
    }
    __syncthreads();
    float4 reg_B;
    reg_B.x = __fdividef(__expf(reg_A.x - block_md.m), block_md.d);
    reg_B.y = __fdividef(__expf(reg_A.y - block_md.m), block_md.d);
    reg_B.z = __fdividef(__expf(reg_A.z - block_md.m), block_md.d);
    reg_B.w = __fdividef(__expf(reg_A.w - block_md.m), block_md.d);
    FLOAT4(thread_B_start[0]) = reg_B;
}

// online safe-softmax
#define LAUNCH_ONLINE_SOFTMAX_F32x4(H) \
    online_softmax_f32x4_kernel<(H) / 4><<<grid, block>>>(reinterpret_cast<float*>(A.data_ptr()), reinterpret_cast<float*>(B.data_ptr()), H);

#define DISPATCH_ONLINE_SOFTMAX_F32x4(S, H)                              \
    dim3 block((H) / 4);                                                 \
    dim3 grid((S));                                                      \
    switch ((H)) {                                                       \
        case 32:                                                         \
            LAUNCH_ONLINE_SOFTMAX_F32x4(32) break;                       \
        case 64:                                                         \
            LAUNCH_ONLINE_SOFTMAX_F32x4(64) break;                       \
        case 128:                                                        \
            LAUNCH_ONLINE_SOFTMAX_F32x4(128) break;                      \
        case 256:                                                        \
            LAUNCH_ONLINE_SOFTMAX_F32x4(256) break;                      \
        case 512:                                                        \
            LAUNCH_ONLINE_SOFTMAX_F32x4(512) break;                      \
        case 1024:                                                       \
            LAUNCH_ONLINE_SOFTMAX_F32x4(1024) break;                     \
        case 2048:                                                       \
            LAUNCH_ONLINE_SOFTMAX_F32x4(2048) break;                     \
        case 4096:                                                       \
            LAUNCH_ONLINE_SOFTMAX_F32x4(4096) break;                     \
        default:                                                         \
            throw std::runtime_error("only support H: 64/128/.../4096"); \
            break;                                                       \
    }

void online_softmax_f32x4(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(A, B)
    const int S = A.size(0);  // seqlens
    const int H = A.size(1);  // head size/kv_len
    // const int N = S * H;
    DISPATCH_ONLINE_SOFTMAX_F32x4(S, H)
}