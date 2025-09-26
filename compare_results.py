#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 读取原版PaddleOCR结果
with open('test/pic01.jpg.txt', 'r', encoding='utf-8') as f:
    original_lines = [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-')]

# 读取ONNX版本结果  
with open('test/pic01.jpg.onnx.txt', 'r', encoding='utf-8') as f:
    onnx_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

print(f"📊 文本提取对比分析")
print(f"=" * 50)
print(f"原版PaddleOCR提取行数: {len(original_lines)}")
print(f"ONNX版本提取行数: {len(onnx_lines)}")
print(f"缺失行数: {len(original_lines) - len(onnx_lines)}")
print(f"缺失比例: {(len(original_lines) - len(onnx_lines)) / len(original_lines) * 100:.1f}%")

print(f"\n📋 原版PaddleOCR完整提取:")
for i, line in enumerate(original_lines, 1):
    print(f"{i:2d}: {line}")

print(f"\n📋 ONNX版本提取:")
for i, line in enumerate(onnx_lines, 1):
    print(f"{i:2d}: {line}")