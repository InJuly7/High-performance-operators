#include "safe_softmax.cuh"
#include "../include/pybind.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(softmax_f32);
    TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4);
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32);
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4);
    TORCH_BINDING_COMMON_EXTENSION(online_softmax_f32);
    TORCH_BINDING_COMMON_EXTENSION(online_softmax_f32x4);

    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16);
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8);
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_pack);
    TORCH_BINDING_COMMON_EXTENSION(online_softmax_f16x8);

}