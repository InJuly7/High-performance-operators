#!/bin/bash

# 检查是否提供了 .cu 文件参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <filename.cu>"
    echo "示例: $0 ./sgemm_v0_global_memory.cu"
    exit 1
fi

# 获取第一个参数
CU_FILE=$1

# 检查 .cu 文件是否存在
if [ ! -f "$CU_FILE" ]; then
    echo "错误: 文件 '$CU_FILE' 不存在"
    exit 1
fi



# 获取文件名（不含扩展名）作为可执行文件名
BASENAME="$(basename "$CU_FILE" .cu)"
OUTPUT_DIR="./bin"
INCLUDE_DIR="./include"

CUDA_FLAGS="-arch=sm_60"
CXX_FLAGS="-std=c++11"
EXECUTABLE="$OUTPUT_DIR/$BASENAME"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 编译 CUDA 程序
NVCC_CMD="nvcc -I$INCLUDE_DIR $CUDA_FLAGS $CXX_FLAGS -o $EXECUTABLE $CU_FILE"
echo "编译命令: $NVCC_CMD"
$NVCC_CMD

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功! 正在运行..."
    echo "----------------------------------------"
    "$EXECUTABLE"
    echo "----------------------------------------"
    echo "运行完成"
    # 可选：运行完成后删除可执行文件
    # rm "$EXECUTABLE"
else
    echo "编译失败!"
    exit 1
fi

