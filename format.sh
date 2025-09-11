#!/bin/bash
echo "正在格式化 C++ 文件..."
find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.c" -o -name "*.cxx" -o -name "*.h" -o -name "*.cuh" -o -name "*.hpp" \) -exec echo "格式化: {}" \; -exec clang-format -i {} \;
echo "完成！"