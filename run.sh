#!/bin/bash

# 检查是否传入参数
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh --sm_<compute_capability> <filename>"
    echo "Example: ./run.sh --sm_75 main.cu"
    exit 1
fi

# 获取架构参数
if [[ "$1" =~ --sm_([0-9]+) ]]; then
    ARCH="sm_${BASH_REMATCH[1]}"
else
    echo "Invalid argument. Use format: --sm_<compute_capability> (e.g., --sm_61)"
    exit 1
fi

# 获取文件名
FILENAME="$2"

# 检查文件是否存在
if [ ! -f "$FILENAME" ]; then
    echo "Error: File '$FILENAME' not found."
    exit 1
fi

# 提取文件名（不包含扩展名）作为输出文件名
OUTPUT_NAME=$(basename "$FILENAME" .cu)

# 编译 CUDA 程序
nvcc -arch=${ARCH} -O3 "$FILENAME" -o "$OUTPUT_NAME"
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful: $FILENAME -> $OUTPUT_NAME"

# # 根据架构执行不同的命令
# if [[ "$ARCH" == "sm_61" ]]; then
#     ./"$OUTPUT_NAME"
#     rm "$OUTPUT_NAME"
# elif [[ "$ARCH" == "sm_75" ]]; then
#     ncu -o kernel_profile --set full -f ./"$OUTPUT_NAME"
# else
#     echo "Unsupported architecture for additional actions."
# fi
