#!/usr/bin/env python3
"""
OCR结果结构测试脚本
用于调试OCR返回结果的数据结构
"""
import sys
sys.path.append('.')

from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path

def test_ocr_structure():
    """测试OCR结果结构"""
    print("初始化PaddleOCR...")
    ocr = PaddleOCR(lang='ch')
    
    # 创建一个简单的测试图片（白底黑字）
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'Hello World', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("执行OCR...")
    result = ocr.ocr(img)
    
    print(f"OCR结果类型: {type(result)}")
    print(f"OCR结果长度: {len(result) if result else 'None'}")
    
    if result:
        ocr_result = result[0]
        print(f"result[0]类型: {type(ocr_result)}")
        print(f"result[0]完整类型名: {str(type(ocr_result))}")
        
        # 检查是否是OCRResult对象
        if str(type(ocr_result)).find('OCRResult') != -1:
            print("这是一个OCRResult对象")
            
            # 列出所有属性
            attrs = [attr for attr in dir(ocr_result) if not attr.startswith('_')]
            print(f"可用属性: {attrs}")
            
            # 检查常见的文本属性
            for attr in ['rec_texts', 'rec_scores', 'text', 'texts', 'content']:
                if hasattr(ocr_result, attr):
                    value = getattr(ocr_result, attr)
                    print(f"{attr}: {type(value)} = {value}")
            
            # 由于它有字典方法，尝试作为字典访问
            if hasattr(ocr_result, 'keys'):
                print(f"字典键: {list(ocr_result.keys())}")
                
                # 打印所有键值对
                for key in ocr_result.keys():
                    value = ocr_result[key]
                    print(f"  {key}: {type(value)} = {value}")
                    
                # 检查常见的文本键
                for text_key in ['rec_texts', 'texts', 'text', 'content', 'result']:
                    if text_key in ocr_result:
                        texts = ocr_result[text_key]
                        print(f"找到文本数据在 '{text_key}': {texts}")
            
            # 尝试访问rec_texts
            if hasattr(ocr_result, 'rec_texts'):
                texts = ocr_result.rec_texts
                print(f"rec_texts内容: {texts}")
                print(f"rec_texts类型: {type(texts)}")
                
                if hasattr(ocr_result, 'rec_scores'):
                    scores = ocr_result.rec_scores
                    print(f"rec_scores内容: {scores}")
                    print(f"rec_scores类型: {type(scores)}")
        else:
            print("这不是OCRResult对象，尝试传统格式...")
            if hasattr(ocr_result, '__iter__'):
                for i, item in enumerate(ocr_result):
                    print(f"Item {i}: {type(item)} = {item}")

if __name__ == "__main__":
    test_ocr_structure()
