#!/usr/bin/env python3
"""
示例脚本：演示如何使用 OCR 性能测试工具的 API

这个脚本展示了如何直接使用 OCRPerformanceTester 类，
而不是通过命令行界面。
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径，以便导入 ocr_performance 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_performance import OCRPerformanceTester


def example_single_image():
    """单张图片处理示例"""
    print("=== 单张图片处理示例 ===")
    
    # 初始化 OCR 测试器
    tester = OCRPerformanceTester(
        use_angle_cls=True,
        lang='ch',  # 中文识别
        use_gpu=False  # 使用 CPU + OpenVINO
    )
    
    # 假设有一张测试图片
    image_path = "test_image.jpg"
    
    if not Path(image_path).exists():
        print(f"测试图片不存在: {image_path}")
        print("请将测试图片命名为 test_image.jpg 并放在当前目录")
        return
    
    try:
        # 提取文本并测量性能
        text, processing_time = tester.extract_text_from_image(image_path)
        
        print(f"图片: {image_path}")
        print(f"处理时间: {processing_time:.4f} 秒")
        print(f"提取的文本长度: {len(text)} 字符")
        print(f"提取的文本预览: {text[:100]}...")
        
        # 保存结果
        tester.save_text_result(image_path, text, processing_time)
        print(f"结果已保存到: {image_path}.txt")
        
    except Exception as e:
        print(f"处理失败: {e}")


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 初始化 OCR 测试器
    tester = OCRPerformanceTester(
        use_angle_cls=True,
        lang='ch',
        use_gpu=False
    )
    
    # 假设有一个图片目录
    images_dir = "sample_images"
    
    if not Path(images_dir).exists():
        print(f"测试目录不存在: {images_dir}")
        print("请创建 sample_images 目录并放入一些测试图片")
        return
    
    try:
        # 批量处理
        successful, total, total_time = tester.process_directory(images_dir)
        
        print(f"\n处理结果:")
        print(f"成功处理: {successful}/{total} 张图片")
        print(f"总处理时间: {total_time:.4f} 秒")
        
        if successful > 0:
            avg_time = total_time / successful
            print(f"平均处理时间: {avg_time:.4f} 秒/张")
        
    except Exception as e:
        print(f"批量处理失败: {e}")


def example_performance_comparison():
    """性能对比示例"""
    print("\n=== 性能对比示例 ===")
    
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"测试图片不存在: {image_path}")
        return
    
    configs = [
        {"name": "标准配置", "use_angle_cls": True, "lang": "ch"},
        {"name": "快速配置", "use_angle_cls": False, "lang": "ch"},
        {"name": "英文配置", "use_angle_cls": True, "lang": "en"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        
        try:
            tester = OCRPerformanceTester(
                use_angle_cls=config["use_angle_cls"],
                lang=config["lang"],
                use_gpu=False
            )
            
            text, processing_time = tester.extract_text_from_image(image_path)
            
            results.append({
                "config": config["name"],
                "time": processing_time,
                "text_length": len(text)
            })
            
            print(f"  处理时间: {processing_time:.4f} 秒")
            print(f"  文本长度: {len(text)} 字符")
            
        except Exception as e:
            print(f"  配置测试失败: {e}")
    
    # 显示对比结果
    if results:
        print(f"\n=== 性能对比结果 ===")
        fastest = min(results, key=lambda x: x["time"])
        
        for result in results:
            speed_ratio = result["time"] / fastest["time"]
            print(f"{result['config']}: {result['time']:.4f}s "
                  f"(相对最快 {speed_ratio:.2f}x)")


def main():
    """主函数"""
    print("PaddleOCR 性能测试工具 - API 使用示例")
    print("=" * 50)
    
    # 检查当前目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 运行示例
    try:
        example_single_image()
        example_batch_processing()
        example_performance_comparison()
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n程序异常: {e}")
    
    print("\n示例运行完成！")


if __name__ == "__main__":
    main()
