#pragma once
#include "../include/pybind.hpp"

void rms_norm_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void rms_norm_f32x4(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void rms_norm_f16(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void rms_norm_f16x2(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void rms_norm_f16x8(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void rms_norm_f16_pack(torch::Tensor A, torch::Tensor B, torch::Tensor C);