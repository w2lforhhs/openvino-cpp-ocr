# PaddleOCR OpenVINO Backend Tool

这个工具使用OpenVINO后端运行PaddleOCR模型，提供高性能的OCR文本提取功能。支持处理单个图像文件或整个目录，并将提取的文本保存为带有`.openvino.txt`后缀的文件。

## 功能特点

- ✅ **OpenVINO加速**: 使用OpenVINO后端优化推理性能
- 📁 **批量处理**: 支持单个文件或整个目录的批量处理
- 🌐 **多语言支持**: 支持中文、英文等多种语言
- 📱 **多设备支持**: 支持CPU、GPU和AUTO设备选择
- 🔄 **角度分类**: 可选的文字方向检测和校正
- 💾 **自动保存**: 结果自动保存为`.openvino.txt`文件

## 安装要求

确保已安装以下依赖包：

```bash
pip install paddleocr openvino opencv-python Pillow numpy tqdm
```

## 使用方法

### 基本用法

```bash
# 处理单个图像文件
python ocr_openvino.py image.jpg

# 处理目录中的所有图像
python ocr_openvino.py ./images/
```

### 高级选项

```bash
# 使用英文识别
python ocr_openvino.py image.jpg --lang en

# 使用GPU设备（如果可用）
python ocr_openvino.py image.jpg --device GPU

# 禁用角度分类（更快的处理速度）
python ocr_openvino.py image.jpg --no-angle-cls

# 组合选项
python ocr_openvino.py ./images/ --lang en --device CPU --no-angle-cls
```

### 参数说明

- `input_path`: 图像文件路径或包含图像的目录路径
- `--lang`: 识别语言 (默认: ch)
  - `ch`: 中文
  - `en`: 英文
  - `fr`: 法文
  - `german`: 德文
  - `korean`: 韩文
  - `japan`: 日文
- `--device`: OpenVINO设备 (默认: CPU)
  - `CPU`: 使用CPU
  - `GPU`: 使用GPU（需要GPU支持）
  - `AUTO`: 自动选择最佳设备
- `--no-angle-cls`: 禁用角度分类（提高速度但可能降低旋转文字的准确性）
- `--gpu`: GPU加速的兼容性选项

## 输出格式

工具会为每个处理的图像创建一个对应的文本文件：

```
原始文件:    image.jpg
输出文件:    image.jpg.openvino.txt
```

输出文件包含从图像中提取的所有文本，每行一个文本块。

## 示例输出

```bash
🔥 PaddleOCR OpenVINO Backend OCR Tool
==================================================
🚀 Initializing PaddleOCR with OpenVINO backend...
✓ OpenVINO version: 2025.3.0
📱 Available OpenVINO devices: ['CPU']
🔧 OpenVINO environment configured for CPU
✓ PaddleOCR initialized successfully with OpenVINO backend

🎯 Target: test/pic01.jpg
🌐 Language: ch
📱 Device: CPU
--------------------------------------------------
📄 Processing single file: test/pic01.jpg
  🖼️ Processing: pic01.jpg
    ⏱️ Processing time: 4.078s
    📝 Extracted 69 words
  💾 Saved to: pic01.jpg.openvino.txt
--------------------------------------------------
✅ Processing completed successfully!
⏱️ Total execution time: 4.083s
💾 Results saved with .openvino.txt suffix
```

## 性能优化建议

1. **设备选择**:
   - 使用`--device CPU`获得最稳定的性能
   - 如果有OpenVINO GPU支持，可尝试`--device GPU`

2. **角度分类**:
   - 对于正常方向的文档，使用`--no-angle-cls`可以显著提高速度
   - 对于可能包含旋转文字的图像，保持角度分类开启

3. **语言选择**:
   - 根据文档语言选择正确的`--lang`参数可以提高识别准确性

## 支持的图像格式

- JPG/JPEG
- PNG
- BMP
- TIFF/TIF
- WebP

## 故障排除

1. **OpenVINO未安装**:
   ```bash
   pip install openvino
   ```

2. **缺少依赖**:
   ```bash
   pip install -r requirements.txt
   ```

3. **内存不足**:
   - 尝试使用较小的图像
   - 减少并发处理

4. **GPU不可用**:
   - 检查OpenVINO GPU驱动是否正确安装
   - 回退到CPU模式：`--device CPU`

## 与原版对比

| 特性 | ocr_performance.py | ocr_openvino.py |
|------|-------------------|-----------------|
| 后端 | 默认PaddlePaddle | OpenVINO优化 |
| 文件后缀 | `.txt` | `.openvino.txt` |
| 性能 | 标准性能 | OpenVINO加速 |
| 设备支持 | GPU/CPU | CPU/GPU/AUTO |

## 许可证

本工具基于PaddleOCR和OpenVINO，请遵循相应的开源许可证。
