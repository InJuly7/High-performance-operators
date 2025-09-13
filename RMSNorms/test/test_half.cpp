#include <iostream>
#include <cstdlib>  // for malloc/free
#include "half.hpp"

using half_float::half;

int main() {
    // 1. 正确的构造方式
    half a(3.14f);          // 显式构造
    half b(2.5f);
    // 或者使用赋值构造
    half c = half(1.5f);
    
    std::cout << "创建 half 精度数字:" << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "c = " << c << std::endl;
    
    // 2. 基本运算（运算结果可以直接赋值）
    half sum = a + b;       // 运算结果是 half 类型，可以直接赋值
    half product = a * b;
    half quotient = a / b;
    
    std::cout << "\n基本运算:" << std::endl;
    std::cout << "a + b = " << sum << std::endl;
    std::cout << "a * b = " << product << std::endl;
    std::cout << "a / b = " << quotient << std::endl;
    
    // 3. 栈上数组的正确创建方式
    std::cout << "\n栈上数组:" << std::endl;
    
    // 方式1: 显式构造每个元素
    half stack_array1[5] = {half(1.1f), half(2.2f), half(3.3f), half(4.4f), half(5.5f)};
    
    // 方式2: 先创建再赋值
    half stack_array2[3];
    stack_array2[0] = half(10.0f);
    stack_array2[1] = half(20.0f);
    stack_array2[2] = half(30.0f);
    
    std::cout << "栈数组1: ";
    for (int i = 0; i < 5; i++) {
        std::cout << stack_array1[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "栈数组2: ";
    for (int i = 0; i < 3; i++) {
        std::cout << stack_array2[i] << " ";
    }
    std::cout << std::endl;
    
    // 4. 使用 malloc 创建堆上数组
    std::cout << "\n堆上数组 (malloc):" << std::endl;
    
    size_t array_size = 10;
    
    // 分配内存
    half* heap_array = (half*)malloc(array_size * sizeof(half));
    
    if (heap_array == nullptr) {
        std::cerr << "内存分配失败!" << std::endl;
        return 1;
    }
    
    // 初始化数组 - 需要使用 placement new 或直接赋值
    for (size_t i = 0; i < array_size; i++) {
        // 方式1: 使用 placement new 显式构造
        // new (&heap_array[i]) half(static_cast<float>(i) * 1.5f);
        
        // 方式2: 直接赋值（如果已经构造过）
        heap_array[i] = half(static_cast<float>(i) * 1.5f);
    }
    
    std::cout << "堆数组 (malloc): ";
    for (size_t i = 0; i < array_size; i++) {
        std::cout << heap_array[i] << " ";
    }
    std::cout << std::endl;
    
    // 计算数组和
    half heap_sum(0.0f);
    for (size_t i = 0; i < array_size; i++) {
        heap_sum = heap_sum + heap_array[i];  // 或 heap_sum += heap_array[i];
    }
    std::cout << "堆数组总和: " << heap_sum << std::endl;
    
    // 释放内存（对于 POD 类型，half 通常不需要显式析构）
    free(heap_array);
    
    // 5. 更推荐的方式：使用 new/delete
    std::cout << "\n堆上数组 (new/delete):" << std::endl;
    
    half* new_array = new half[array_size];
    
    // 初始化
    for (size_t i = 0; i < array_size; i++) {
        new_array[i] = half(static_cast<float>(i) * 2.0f);
    }
    
    std::cout << "new 数组: ";
    for (size_t i = 0; i < array_size; i++) {
        std::cout << new_array[i] << " ";
    }
    std::cout << std::endl;
    
    delete[] new_array;
    
    // 6. 使用 calloc 的例子
    std::cout << "\n使用 calloc:" << std::endl;
    
    half* calloc_array = (half*)calloc(array_size, sizeof(half));
    
    if (calloc_array != nullptr) {
        // calloc 会将内存初始化为0，但对于 half 类型，我们仍需要显式初始化
        for (size_t i = 0; i < array_size; i++) {
            new (&calloc_array[i]) half(static_cast<float>(i) + 0.5f);
        }
        
        std::cout << "calloc 数组: ";
        for (size_t i = 0; i < array_size; i++) {
            std::cout << calloc_array[i] << " ";
        }
        std::cout << std::endl;
        
        free(calloc_array);
    }
    
    // 7. 内存大小验证
    std::cout << "\n内存大小验证:" << std::endl;
    std::cout << "sizeof(half): " << sizeof(half) << " 字节" << std::endl;
    std::cout << "10个half的内存: " << 10 * sizeof(half) << " 字节" << std::endl;
    std::cout << "10个float的内存: " << 10 * sizeof(float) << " 字节" << std::endl;
    
    return 0;
}