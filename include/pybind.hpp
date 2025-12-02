#pragma once
#include <torch/extension.h>
#include <torch/types.h>

// 简化 PyTorch C++ 扩展（Extension）的算子绑定
// 将传入宏的参数直接转换成一个 C 风格的字符串常量（即在两头加上双引号）
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
    if (((T).options().dtype() != (th_type))) {                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type);      \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                       \
    assert((T1).dim() == (T2).dim());                          \
    for (int i = 0; i < (T1).dim(); ++i) {                     \
        if ((T2).size(i) != (T1).size(i)) {                    \
            throw std::runtime_error("Tensor size mismatch!"); \
        }                                                      \
    }
