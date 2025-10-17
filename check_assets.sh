#!/bin/bash
# 检查PLY文件是否正确下载

echo "检查资产文件状态..."
echo ""

for ply_file in assets/example_assert/object/*.ply assets/example_assert/scene/*.ply; do
    if [ -f "$ply_file" ]; then
        first_line=$(head -1 "$ply_file")
        size=$(stat -f%z "$ply_file" 2>/dev/null || stat -c%s "$ply_file" 2>/dev/null)

        if [ "$first_line" = "ply" ]; then
            echo "✓ $ply_file - 正确的PLY文件 ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo ${size}B))"
        elif [[ "$first_line" == *"git-lfs"* ]]; then
            echo "✗ $ply_file - git-lfs指针文件，需要运行 'git lfs pull'"
        else
            echo "? $ply_file - 未知格式"
        fi
    fi
done

echo ""
echo "如果看到✗标记，请运行："
echo "  git lfs install"
echo "  git lfs pull"
