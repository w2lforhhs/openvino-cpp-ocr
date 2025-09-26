# PaddleOCR v4 + OpenVINO 性能测试工具

一个用于测试 PaddleOCR v4 与 OpenVINO 加速的图片 OCR 提取性能的命令行工具。

## 功能特性

- ✅ **PaddleOCR v4 集成**: 支持最新版本的 PaddleOCR
- ⚡ **OpenVINO 加速**: 使用 Intel OpenVINO 进行 CPU 优化
- 🖼️ **灵活输入**: 支持单个图片文件或整个目录批量处理
- 📝 **自动保存**: 提取的文本自动保存为 `.txt` 文件（原文件名 + .txt）
- ⏱️ **性能统计**: 详细记录每张图片的处理耗时
- 🌍 **多语言支持**: 支持中文、英文等多种语言识别
- 🔧 **环境隔离**: 提供一键部署脚本，自动创建虚拟环境

## 快速开始

### 1. 一键环境部署

**Windows:**
```cmd
# 双击运行或在命令行执行
setup.bat
```

**Linux/macOS:**
```bash
# 给脚本执行权限并运行
chmod +x setup.sh
./setup.sh
```

### 2. 激活环境

**Windows:**
```cmd
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. 运行 OCR 测试

```bash
# 处理单个图片
python ocr_performance.py image.jpg

# 处理整个目录
python ocr_performance.py /path/to/images/

# 使用英文识别
python ocr_performance.py image.png --lang en

# 使用 GPU 加速（需要 NVIDIA GPU）
python ocr_performance.py images/ --gpu

# 禁用角度分类以提升速度
python ocr_performance.py image.jpg --no-angle-cls
```

## 使用示例

### 单图片处理
```bash
python ocr_performance.py test_image.jpg
```

输出示例：
```
Initializing PaddleOCR with OpenVINO backend...
✓ PaddleOCR initialized successfully (GPU: False, Language: ch)

🖼️  Processing single image: test_image.jpg
Processing: test_image.jpg
  ⏱️  Processing time: 1.2345 seconds
  📝 Extracted 156 characters
✓ Text saved to: test_image.jpg.txt

✅ Successfully processed image

⏱️  Total elapsed time: 1.3456 seconds
✨ Processing complete!
```

### 批量目录处理
```bash
python ocr_performance.py images_folder/
```

输出示例：
```
📁 Processing directory: images_folder/
Found 10 image files in directory

Processing images: 100%|████████████| 10/10 [00:15<00:00,  1.54s/it]
  ✓ image1.jpg: 1.2345s, 123 chars
  ✓ image2.png: 0.9876s, 89 chars
  ...

📊 Processing Summary:
   Successful: 10/10 images
   Total OCR time: 12.3456 seconds
   Average time per image: 1.2346 seconds

⏱️  Total elapsed time: 15.6789 seconds
✨ Processing complete!
```

## 输出格式

每张图片处理后会生成对应的 `.txt` 文件，包含：

```
# OCR Results for: example.jpg
# Processing time: 1.2345 seconds
# Timestamp: 2025-09-09 10:30:45
--------------------------------------------------

这里是提取的文本内容
支持多行文本
包含所有识别出的文字
```

## 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `input_path` | 图片文件路径或包含图片的目录 | 必需 |
| `--lang` | OCR 识别语言 (`ch`, `en` 等) | `ch` |
| `--gpu` | 启用 GPU 加速 | 否 |
| `--no-angle-cls` | 禁用角度分类（提升速度） | 否 |

## 支持的图片格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10+, Linux, macOS
- **内存**: 建议 4GB 以上
- **存储**: 约 2GB 用于模型文件下载

### GPU 加速 (可选)
- **NVIDIA GPU**: 支持 CUDA 的显卡
- **显存**: 建议 2GB 以上

## 性能优化建议

1. **CPU 优化**: 
   - 工具已默认启用 Intel MKL-DNN 优化
   - 自动使用所有可用 CPU 核心

2. **GPU 加速**: 
   - 使用 `--gpu` 参数启用
   - 需要安装 `paddlepaddle-gpu`

3. **角度分类**: 
   - 对于不含旋转文字的图片，可使用 `--no-angle-cls` 提升速度

4. **批量处理**: 
   - 目录批量处理比单个文件处理更高效

## 故障排除

### 常见问题

**1. 依赖安装失败**
```bash
# 更新 pip
python -m pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**2. 模型下载缓慢**
```bash
# 设置环境变量使用国内镜像
export HUB_HOME=~/.paddlehub
```

**3. GPU 不可用**
```bash
# 安装 GPU 版本的 PaddlePaddle
pip install paddlepaddle-gpu
```

**4. 内存不足**
```bash
# 减少并发处理，一次处理一张图片
python ocr_performance.py single_image.jpg
```

## 目录结构

```
paddleocr-performance/
├── .github/
│   └── copilot-instructions.md    # 项目指导文档
├── ocr_performance.py             # 主程序
├── requirements.txt               # 基础依赖
├── requirements-detailed.txt      # 详细版本依赖
├── setup.bat                      # Windows 一键部署
├── setup.sh                       # Linux/macOS 一键部署
└── README.md                      # 项目文档
```

## 开发说明

### 添加新功能
1. 克隆项目到本地
2. 创建虚拟环境并安装依赖
3. 在 `ocr_performance.py` 中添加功能
4. 更新文档和测试

### 自定义配置
可以修改 `OCRPerformanceTester` 类的初始化参数来调整 OCR 行为：

```python
# 自定义 OCR 配置
tester = OCRPerformanceTester(
    use_angle_cls=True,      # 角度分类
    lang='en',               # 语言
    use_gpu=False           # GPU 使用
)
```

## 许可证

本项目基于 MIT 许可证开源。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

## 更新日志

### v1.0.0 (2025-09-09)
- ✨ 初始版本发布
- ✅ 支持 PaddleOCR v4 + OpenVINO
- ✅ 命令行界面
- ✅ 批量处理功能
- ✅ 性能统计
- ✅ 一键部署脚本
