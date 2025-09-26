#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR v4 + OpenVINO Performance Testing Tool

This tool provides OCR performance testing using PaddleOCR v4 with OpenVINO acceleration.
It supports processing single images or entire directories, measuring extraction time,
and saving results to .txt files alongside the original images.

支持中文路径和文件名的OCR性能测试工具
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from tqdm import tqdm


class OCRPerformanceTester:
    """PaddleOCR performance testing with OpenVINO acceleration"""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'ch', use_gpu: bool = False):
        """
        Initialize PaddleOCR with OpenVINO backend
        
        Args:
            use_angle_cls: Whether to use angle classification (mapped to use_textline_orientation)
            lang: Language for OCR ('ch', 'en', etc.)
            use_gpu: Whether to use GPU (if available)
        """
        print("Initializing PaddleOCR with OpenVINO backend...")
        
        # Check OpenVINO availability
        self._check_openvino_status()
        
        try:
            # PaddleOCR initialization with updated parameters
            init_params = {
                'use_textline_orientation': use_angle_cls,  # Updated parameter name
                'lang': lang
            }
            
            # Note: PaddleOCR v4 might not support use_gpu parameter directly
            # GPU usage is typically controlled by environment variables or paddle settings
            if use_gpu:
                print("⚠️  GPU mode requested but may require additional paddle configuration")
            
            self.ocr = PaddleOCR(**init_params)
            print(f"✓ PaddleOCR initialized successfully (GPU: {use_gpu}, Language: {lang})")
            
            # Check backend information after initialization
            self._check_backend_info()
            
        except Exception as e:
            print(f"✗ Failed to initialize PaddleOCR: {e}")
            sys.exit(1)
    
    def _check_openvino_status(self):
        """Check if OpenVINO is available and properly configured"""
        print("🔍 Checking OpenVINO status...")
        
        # Check if OpenVINO package is installed
        try:
            import openvino as ov
            openvino_version = ov.__version__
            print(f"✓ OpenVINO installed: v{openvino_version}")
        except ImportError:
            print("⚠️  OpenVINO package not found - will use default backend")
            return False
        
        # Check available devices
        try:
            core = ov.Core()
            available_devices = core.available_devices
            print(f"📱 Available OpenVINO devices: {available_devices}")
            
            # Check CPU capabilities
            if 'CPU' in available_devices:
                cpu_device_info = core.get_property('CPU', 'FULL_DEVICE_NAME')
                print(f"🖥️  CPU device: {cpu_device_info}")
        except Exception as e:
            print(f"⚠️  Error checking OpenVINO devices: {e}")
        
        return True
    
    def _check_backend_info(self):
        """Check the actual backend being used by PaddleOCR"""
        print("🔍 Checking PaddleOCR backend configuration...")
        
        try:
            # Try to access internal configuration if available
            if hasattr(self.ocr, 'text_detector') and hasattr(self.ocr.text_detector, 'predictor'):
                detector_config = getattr(self.ocr.text_detector.predictor, 'config', None)
                if detector_config:
                    print(f"📋 Text detector config available")
            
            if hasattr(self.ocr, 'text_recognizer') and hasattr(self.ocr.text_recognizer, 'predictor'):
                recognizer_config = getattr(self.ocr.text_recognizer.predictor, 'config', None)
                if recognizer_config:
                    print(f"📋 Text recognizer config available")
                    
            # Check if using GPU or CPU
            import paddle
            if paddle.is_compiled_with_cuda() and paddle.device.get_device():
                device_info = paddle.device.get_device()
                print(f"🎮 Paddle device: {device_info}")
            else:
                print(f"🖥️  Using CPU backend (OpenVINO optimization expected)")
                
        except Exception as e:
            print(f"⚠️  Could not determine backend details: {e}")
        
        print("✨ Backend check complete")
    
    def extract_text_from_image(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """
        Extract text from a single image and measure performance
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text, processing_time_seconds)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and preprocess image - support Chinese paths
            # Use cv2.imdecode to handle Chinese file paths
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Start timing
            start_time = time.time()
            
            # Perform OCR - updated API without cls parameter
            result = self.ocr.ocr(img)
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Extract text from results with robust error handling
            extracted_text = ""
            print(f"  🔍 OCR result structure: {type(result)}, length: {len(result) if result else 'None'}")
            
            if result and len(result) > 0:
                ocr_result = result[0]
                print(f"  🔍 result[0] type: {type(ocr_result)}")
                
                # Handle OCRResult object - check dictionary access first
                if str(type(ocr_result)).find('OCRResult') != -1:
                    print(f"  📝 Handling OCRResult object as dictionary")
                    
                    # OCRResult is a dictionary-like object, access rec_texts directly
                    if 'rec_texts' in ocr_result:
                        texts = ocr_result['rec_texts']
                        scores = ocr_result.get('rec_scores', [])
                        print(f"  📝 Found rec_texts with {len(texts)} items")
                        
                        for i, text in enumerate(texts):
                            score = scores[i] if i < len(scores) else 0.0
                            print(f"    📖 Text {i+1}: '{text}' (confidence: {score:.4f})")
                            extracted_text += f"{text}\n"
                    else:
                        # Print available keys for debugging
                        if hasattr(ocr_result, 'keys'):
                            keys = list(ocr_result.keys())
                            print(f"  🔍 Available keys: {keys}")
                
                elif hasattr(ocr_result, 'rec_texts'):
                    # Fallback: try attribute access
                    print(f"  📝 Using attribute access for rec_texts")
                    texts = ocr_result.rec_texts
                    scores = getattr(ocr_result, 'rec_scores', [])
                    
                    print(f"  📝 Found {len(texts)} text items")
                    for i, text in enumerate(texts):
                        score = scores[i] if i < len(scores) else 0.0
                        print(f"    📖 Text {i+1}: '{text}' (confidence: {score:.4f})")
                        extracted_text += f"{text}\n"
                        
                elif hasattr(ocr_result, '__iter__') and not isinstance(ocr_result, str):
                    # Legacy format handling
                    print(f"  📝 Using legacy format")
                    try:
                        for i, line in enumerate(ocr_result):
                            print(f"    Line {i+1} structure: {type(line)}, content: {line}")
                            # Check if line has the expected structure
                            if isinstance(line, (list, tuple)) and len(line) >= 2:
                                # line[0] should be coordinates, line[1] should be (text, confidence)
                                if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                                    text = str(line[1][0])  # Get the text content safely
                                    confidence = line[1][1] if len(line[1]) > 1 else 0.0
                                    print(f"    📖 Text: '{text}' (confidence: {confidence})")
                                    extracted_text += f"{text}\n"
                                elif isinstance(line[1], str):
                                    # Sometimes the structure might be different
                                    print(f"    📖 Text (direct): '{line[1]}'")
                                    extracted_text += f"{line[1]}\n"
                    except Exception as e:
                        print(f"  ⚠️  Error in legacy format handling: {e}")
                        
                else:
                    print(f"  ⚠️  Unknown result format")
            else:
                print("  ⚠️  No OCR results found")
            
            print(f"  📊 Total extracted text length: {len(extracted_text.strip())} characters")
            
            return extracted_text.strip(), processing_time
            
        except Exception as e:
            raise RuntimeError(f"OCR processing failed for {image_path}: {e}")
    
    def save_text_result(self, image_path: Union[str, Path], text: str, processing_time: float):
        """
        Save extracted text to a .txt file alongside the original image
        
        Args:
            image_path: Path to the original image
            text: Extracted text content
            processing_time: Time taken for processing
        """
        image_path = Path(image_path)
        txt_path = image_path.with_suffix(image_path.suffix + '.txt')
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"# OCR Results for: {image_path.name}\n")
                f.write(f"# Processing time: {processing_time:.4f} seconds\n")
                f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n\n")
                f.write(text)
            
            print(f"✓ Text saved to: {txt_path}")
            
        except Exception as e:
            print(f"✗ Failed to save text file for {image_path}: {e}")
    
    def get_supported_image_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Get list of supported image files in a directory
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of image file paths
        """
        directory = Path(directory)
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        print(f"🔍 Searching for image files in: {directory}")
        
        image_files = []
        
        # 使用 iterdir() 方法来更好地处理中文路径
        try:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    # 检查文件扩展名（不区分大小写）
                    file_suffix = file_path.suffix.lower()
                    if file_suffix in supported_extensions:
                        image_files.append(file_path)
                        print(f"  ✓ Found: {file_path.name}")
        except Exception as e:
            print(f"  ✗ Error reading directory: {e}")
            # 如果 iterdir() 失败，尝试使用 glob
            try:
                for ext in supported_extensions:
                    image_files.extend(directory.glob(f"*{ext}"))
                    image_files.extend(directory.glob(f"*{ext.upper()}"))
            except Exception as e2:
                print(f"  ✗ Glob also failed: {e2}")
        
        return sorted(list(set(image_files)))  # 去重并排序
    
    def process_single_image(self, image_path: Union[str, Path]) -> bool:
        """
        Process a single image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing: {image_path}")
            text, processing_time = self.extract_text_from_image(image_path)
            
            print(f"  ⏱️  Processing time: {processing_time:.4f} seconds")
            print(f"  📝 Extracted {len(text)} characters")
            
            self.save_text_result(image_path, text, processing_time)
            return True
            
        except Exception as e:
            print(f"✗ Error processing {image_path}: {e}")
            return False
    
    def process_directory(self, directory_path: Union[str, Path]) -> Tuple[int, int, float]:
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to the directory containing images
            
        Returns:
            Tuple of (successful_count, total_count, total_time)
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        image_files = self.get_supported_image_files(directory_path)
        
        if not image_files:
            print(f"No supported image files found in: {directory_path}")
            return 0, 0, 0.0
        
        print(f"Found {len(image_files)} image files in directory")
        
        # 调试信息：列出找到的文件
        if image_files:
            print("📋 Found image files:")
            for i, img_file in enumerate(image_files[:10]):  # 只显示前10个
                print(f"  {i+1}. {img_file.name}")
            if len(image_files) > 10:
                print(f"  ... and {len(image_files) - 10} more files")
        
        successful_count = 0
        total_time = 0.0
        
        # Process images with progress bar
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                text, processing_time = self.extract_text_from_image(image_file)
                total_time += processing_time
                
                self.save_text_result(image_file, text, processing_time)
                successful_count += 1
                
                print(f"  ✓ {image_file.name}: {processing_time:.4f}s, {len(text)} chars")
                
            except Exception as e:
                print(f"  ✗ {image_file.name}: {e}")
        
        return successful_count, len(image_files), total_time


def main():
    """Main entry point for the OCR performance testing tool"""
    parser = argparse.ArgumentParser(
        description="PaddleOCR v4 + OpenVINO Performance Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocr_performance.py image.jpg                    # Process single image
  python ocr_performance.py /path/to/images/             # Process directory
  python ocr_performance.py image.png --lang en          # Use English OCR
  python ocr_performance.py images/ --gpu                # Use GPU acceleration
        """
    )
    
    parser.add_argument(
        'input_path',
        help='Path to image file or directory containing images'
    )
    
    parser.add_argument(
        '--lang',
        default='ch',
        help='OCR language (default: ch for Chinese, also supports: en, etc.)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (default: CPU with OpenVINO)'
    )
    
    parser.add_argument(
        '--no-angle-cls',
        action='store_true',
        help='Disable angle classification for faster processing'
    )
    
    args = parser.parse_args()
    
    # Validate input path - support Chinese paths
    try:
        input_path = Path(args.input_path)
        if not input_path.exists():
            print(f"✗ 错误：输入路径不存在: {input_path}")
            print(f"✗ Error: Input path does not exist: {input_path}")
            sys.exit(1)
        
        # Print path info for debugging
        print(f"📂 输入路径: {input_path}")
        print(f"📂 Input path: {input_path}")
        
    except Exception as e:
        print(f"✗ 路径处理错误: {e}")
        print(f"✗ Path processing error: {e}")
        sys.exit(1)
    
    # Initialize OCR tester
    try:
        tester = OCRPerformanceTester(
            use_angle_cls=not args.no_angle_cls,
            lang=args.lang,
            use_gpu=args.gpu
        )
    except Exception as e:
        print(f"✗ Failed to initialize OCR tester: {e}")
        sys.exit(1)
    
    # Process input
    start_time = time.time()
    
    if input_path.is_file():
        # Process single image
        print(f"\n🖼️  Processing single image: {input_path}")
        success = tester.process_single_image(input_path)
        
        if success:
            print(f"\n✅ Successfully processed image")
        else:
            print(f"\n❌ Failed to process image")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Process directory
        print(f"\n📁 Processing directory: {input_path}")
        successful, total, total_processing_time = tester.process_directory(input_path)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Successful: {successful}/{total} images")
        print(f"   Total OCR time: {total_processing_time:.4f} seconds")
        if successful > 0:
            print(f"   Average time per image: {total_processing_time/successful:.4f} seconds")
        
        if successful < total:
            print(f"\n⚠️  {total - successful} images failed to process")
    
    total_elapsed = time.time() - start_time
    print(f"\n⏱️  Total elapsed time: {total_elapsed:.4f} seconds")
    print("✨ Processing complete!")


if __name__ == "__main__":
    main()
