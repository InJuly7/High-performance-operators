#pragma once
#include "../include/pybind.hpp"

// fp32 safe-softmax
void softmax_f32(torch::Tensor A, torch::Tensor B);
void softmax_f32x4(torch::Tensor A, torch::Tensor B);
void safe_softmax_f32(torch::Tensor A, torch::Tensor B);
void safe_softmax_f32x4(torch::Tensor A, torch::Tensor B);
void online_softmax_f32(torch::Tensor A, torch::Tensor B);
void online_softmax_f32x4(torch::Tensor A, torch::Tensor B);

// fp16 safe-softmax
void safe_softmax_f16(torch::Tensor A, torch::Tensor B);
void safe_softmax_f16x8(torch::Tensor A, torch::Tensor B);
void safe_softmax_f16_pack(torch::Tensor A, torch::Tensor B);
void online_softmax_f16x8(torch::Tensor A, torch::Tensor B);