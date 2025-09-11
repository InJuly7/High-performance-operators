#!/bin/bash
echo "正在格式化 C++ 文件..."
find . -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" \) -exec echo "格式化: {}" \; -exec clang-format -i {} \;
echo "完成！"

