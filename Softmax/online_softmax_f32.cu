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
__global__ void online_softmax_f32_kernel(float* mat_A, float* mat_B, int N) {
    float* thread_A_start = mat_A + blockIdx.x * N + threadIdx.x;
    float* thread_B_start = mat_B + blockIdx.x * N + threadIdx.x;
    float thread_A_val = thread_A_start[0];
    const int WARP_NUM = CEIL_DIV(NUM_THREADS, WARP_SIZE);
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    static __shared__ MD warp_MD[WARP_NUM];
    // 默认1.0f, 代表假设每一个元素都是 max, e^{x-m} = 1.0f;
    MD val = {thread_A_val, 1.0f};
    MD warp_val = warp_reduce_md<WARP_SIZE>(val);
    if (laneId == 0) warp_MD[warpId] = warp_val;
    __syncthreads();
    __shared__ MD block_val;
    if (warpId == 0) {
        val = warp_MD[laneId];
        val = warp_reduce_md<WARP_NUM>(val);
        if (laneId == 0) block_val = val;
    }
    __syncthreads();

    thread_B_start[0] = __fdividef(__expf(thread_A_val - block_val.m), block_val.d);
}

// online safe-softmax
#define LAUNCH_ONLINE_SOFTMAX_F32(H) \
    online_softmax_f32_kernel<(H)><<<grid, block>>>(reinterpret_cast<float*>(A.data_ptr()), reinterpret_cast<float*>(B.data_ptr()), H);

#define DISPATCH_ONLINE_SOFTMAX_F32(S, H)                                    \
    dim3 block((H));                                                         \
    dim3 grid((S));                                                          \
    switch ((H)) {                                                           \
        case 32:                                                             \
            LAUNCH_ONLINE_SOFTMAX_F32(32)                                    \
            break;                                                           \
        case 64:                                                             \
            LAUNCH_ONLINE_SOFTMAX_F32(64)                                    \
            break;                                                           \
        case 128:                                                            \
            LAUNCH_ONLINE_SOFTMAX_F32(128)                                   \
            break;                                                           \
        case 256:                                                            \
            LAUNCH_ONLINE_SOFTMAX_F32(256)                                   \
            break;                                                           \
        case 512:                                                            \
            LAUNCH_ONLINE_SOFTMAX_F32(512)                                   \
            break;                                                           \
        case 1024:                                                           \
            LAUNCH_ONLINE_SOFTMAX_F32(1024)                                  \
            break;                                                           \
        default:                                                             \
            throw std::runtime_error("only support H: 64/128/256/512/1024"); \
            break;                                                           \
    }

void online_softmax_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kFloat32)
    CHECK_TORCH_TENSOR_SHAPE(A, B)
    const int S = A.size(0);  // seqlens
    const int H = A.size(1);  // head size/kv_len
    // const int N = S * H;
    DISPATCH_ONLINE_SOFTMAX_F32(S, H)
}