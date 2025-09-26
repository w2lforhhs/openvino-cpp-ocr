#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Performance Comparison Tool
比较原版PaddleOCR、OpenVINO和ONNX+OpenVINO三种方案的性能
"""

import time
import subprocess
import sys
import os
from pathlib import Path

def run_ocr_test(script_name, image_path, description):
    """运行OCR测试并返回执行时间和状态"""
    print(f"\n{'='*60}")
    print(f"🧪 测试方案: {description}")
    print(f"📄 脚本: {script_name}")
    print(f"🖼️ 图像: {image_path}")
    print(f"{'='*60}")
    
    try:
        # 构建命令
        venv_python = "D:/codes/paddleocr-performance/venv/Scripts/python.exe"
        cmd = [venv_python, script_name, image_path]
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行命令
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            timeout=60  # 60秒超时
        )
        
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 检查执行结果
        if result.returncode == 0:
            print(f"✅ 执行成功")
            print(f"⏱️ 执行时间: {elapsed_time:.3f}秒")
            
            # 尝试从输出中提取处理时间
            processing_time = None
            for line in result.stdout.split('\n'):
                if 'Processing time:' in line or 'processing time:' in line:
                    try:
                        time_str = line.split(':')[-1].strip().replace('s', '').replace('秒', '')
                        processing_time = float(time_str)
                    except:
                        pass
                elif 'Total execution time:' in line:
                    try:
                        time_str = line.split(':')[-1].strip().replace('s', '').replace('秒', '')
                        processing_time = float(time_str)
                    except:
                        pass
            
            if processing_time:
                print(f"🔄 OCR处理时间: {processing_time:.3f}秒")
            
            return True, elapsed_time, processing_time or elapsed_time
        else:
            print(f"❌ 执行失败 (错误码: {result.returncode})")
            print(f"⚠️ 错误信息: {result.stderr}")
            return False, elapsed_time, None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 执行超时 (>60秒)")
        return False, 60.0, None
    except Exception as e:
        print(f"💥 执行异常: {e}")
        return False, 0, None

def main():
    """主函数"""
    print("🔥 OCR性能对比测试工具")
    print("=" * 80)
    
    # 测试图像
    test_image = "test/pic01.jpg"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    # 测试方案配置
    test_configs = [
        {
            "script": "ocr_performance.py",
            "description": "原版PaddleOCR v4",
            "args": "--no-angle-cls"
        },
        {
            "script": "ocr_openvino.py", 
            "description": "PaddleOCR + OpenVINO加速",
            "args": "--no-angle-cls"
        },
        {
            "script": "ocr_openvino_onnx.py",
            "description": "ONNX + OpenVINO高性能版",
            "args": "--no-angle-cls"
        }
    ]
    
    # 存储结果
    results = []
    
    # 运行测试
    for i, config in enumerate(test_configs, 1):
        script_name = config["script"]
        description = config["description"]
        args = config.get("args", "")
        
        # 检查脚本文件是否存在
        if not os.path.exists(script_name):
            print(f"⚠️ 跳过测试 {i}: 脚本文件不存在 - {script_name}")
            results.append({
                "name": description,
                "script": script_name,
                "success": False,
                "total_time": 0,
                "processing_time": 0,
                "error": "脚本文件不存在"
            })
            continue
        
        # 构建完整命令
        full_cmd = f"{script_name} {test_image}"
        if args:
            full_cmd += f" {args}"
        
        # 运行测试
        success, total_time, processing_time = run_ocr_test(
            script_name, 
            f"{test_image} {args}".strip(), 
            description
        )
        
        # 记录结果
        results.append({
            "name": description,
            "script": script_name,
            "success": success,
            "total_time": total_time,
            "processing_time": processing_time or total_time,
            "error": None if success else "执行失败"
        })
        
        # 短暂休息
        time.sleep(1)
    
    # 输出汇总结果
    print(f"\n{'='*80}")
    print("📊 性能对比结果汇总")
    print(f"{'='*80}")
    
    # 表头
    print(f"{'方案':<30} {'状态':<8} {'总时间(秒)':<12} {'处理时间(秒)':<12} {'性能提升':<10}")
    print(f"{'-'*80}")
    
    # 找到基准时间（第一个成功的结果）
    baseline_time = None
    for result in results:
        if result["success"] and result["processing_time"]:
            baseline_time = result["processing_time"]
            break
    
    # 输出结果
    for result in results:
        name = result["name"]
        status = "✅成功" if result["success"] else "❌失败"
        total_time = f"{result['total_time']:.3f}" if result["success"] else "N/A"
        processing_time = f"{result['processing_time']:.3f}" if result["success"] and result["processing_time"] else "N/A"
        
        # 计算性能提升
        speedup = "N/A"
        if result["success"] and result["processing_time"] and baseline_time:
            speedup_ratio = baseline_time / result["processing_time"]
            speedup = f"{speedup_ratio:.1f}x"
        
        print(f"{name:<30} {status:<8} {total_time:<12} {processing_time:<12} {speedup:<10}")
    
    # 性能分析
    print(f"\n{'='*80}")
    print("🎯 性能分析结论")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r["success"] and r["processing_time"]]
    if len(successful_results) >= 2:
        # 按处理时间排序
        successful_results.sort(key=lambda x: x["processing_time"])
        
        fastest = successful_results[0]
        slowest = successful_results[-1]
        
        print(f"🥇 最快方案: {fastest['name']}")
        print(f"   ⏱️ 处理时间: {fastest['processing_time']:.3f}秒")
        
        print(f"\n🐌 最慢方案: {slowest['name']}")
        print(f"   ⏱️ 处理时间: {slowest['processing_time']:.3f}秒")
        
        if fastest != slowest:
            speedup = slowest['processing_time'] / fastest['processing_time']
            print(f"\n🚀 性能提升: {speedup:.1f}倍加速")
        
        # 推荐
        print(f"\n💡 推荐使用: {fastest['name']}")
        print(f"   📈 相比传统方案提升 {speedup:.1f}倍性能")
        
    else:
        print("⚠️ 没有足够的成功结果进行比较")
    
    print(f"\n{'='*80}")
    print("🎉 性能测试完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()