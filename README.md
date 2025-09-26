# 🚀 OpenVINO OCR Performance Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.3.0-green.svg)](https://github.com/openvinotoolkit/openvino)

> PaddleOCR v4 + OpenVINO 性能优化项目，实现 **12.1倍** OCR处理速度提升

## ⚡ 核心特性

- 🎯 **极致性能**: ONNX + OpenVINO 实现12.1倍速度提升
- 🔥 **真实模型**: 使用原生ONNX模型 (det_ch.onnx, rec_ch.onnx)
- 🛠️ **智能优化**: 多级fallback策略和质量评估机制
- 🌐 **跨平台**: 支持Windows和Linux环境
- 📊 **完整对比**: 详细的性能分析和质量评估报告

## 📈 性能对比

| 方案 | 处理时间 | 性能提升 | 适用场景 |
|------|----------|----------|----------|
| **ONNX + OpenVINO** | **0.3秒** | **12.1x** | 🚀 高速处理、实时应用 |
| 原版 PaddleOCR v4 | 3.7秒 | 1.0x | 🎯 高精度识别、生产环境 |

## 🚀 快速开始

### 1. 环境设置
```bash
# Windows
setup.bat

# Linux
chmod +x setup.sh && ./setup.sh
```

### 2. 运行对比测试
```bash
# 激活虚拟环境
# Windows: venv\Scripts\activate
# Linux: source venv/bin/activate

# 性能对比
python simple_comparison.py test/pic01.jpg
```

### 3. 单独使用ONNX方案
```bash
python ocr_openvino_onnx.py test/pic01.jpg --no-angle-cls
```

## 📊 技术实现

### 核心优化技术
- **OpenVINO推理引擎**: 原生CPU优化，避免Python解释器开销
- **ONNX模型集成**: 真实的PaddleOCR转换模型
- **字符映射优化**: 使用官方ppocr_keys_v1.txt词典（6623字符）
- **CTC解码增强**: 多级置信度策略和质量评估

### 项目结构
```
openvino-cpp-ocr/
├── 📁 onnx_models/          # ONNX模型文件
├── 📁 test/                 # 测试图片
├── 🚀 ocr_openvino_onnx.py  # 主要优化版本
├── 📊 simple_comparison.py  # 性能对比工具
├── 📋 requirements.txt      # 依赖列表
└── 📚 文档和分析报告/
```

## 📚 详细文档

- 🎯 [最终性能分析](FINAL_PERFORMANCE_ANALYSIS.md) - 完整的技术分析和建议
- 📋 [项目总结](PROJECT_FINAL_SUMMARY.md) - 项目成果和价值分析  
- 🔧 [ONNX使用指南](README_onnx.md) - ONNX版本详细说明
- ⚙️ [OpenVINO设置](README_openvino.md) - OpenVINO环境配置

## 🎯 使用建议

### 🚀 高速场景 (推荐ONNX + OpenVINO)
- 实时视频流处理
- 大批量文档初筛  
- Web服务响应时间敏感应用
- IoT设备轻量级OCR

### 🎯 高精度场景 (推荐原版PaddleOCR)
- 重要文档精确识别
- 代码文档处理
- 法律医疗等高准确性要求
- 面向最终用户的产品功能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

⭐ **如果这个项目对您有帮助，请给个Star支持一下！**