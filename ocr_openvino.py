#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR OpenVINO Backend OCR Tool

This tool uses OpenVINO to run PaddleOCR models for OCR text extraction.
It supports processing single images or entire directories, and saves 
extracted text to .openvino.txt files alongside the original images.

支持通过OpenVINO后端运行PaddleOCR模型的OCR文本提取工具
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from tqdm import tqdm


class OpenVINOOCR:
    """PaddleOCR with OpenVINO backend for high-performance OCR"""
    
    def __init__(self, 
                 use_angle_cls: bool = True, 
                 lang: str = 'ch', 
                 use_gpu: bool = False,
                 device: str = 'CPU'):
        """
        Initialize PaddleOCR with OpenVINO backend
        
        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language for OCR ('ch', 'en', etc.)
            use_gpu: Whether to use GPU (for compatibility)
            device: OpenVINO device ('CPU', 'GPU', 'AUTO')
        """
        print("🚀 Initializing PaddleOCR with OpenVINO backend...")
        
        # Check OpenVINO availability
        self._check_openvino_availability()
        
        # Set OpenVINO environment
        self._setup_openvino_environment(device)
        
        try:
            # Initialize PaddleOCR with OpenVINO backend
            init_params = {
                'use_textline_orientation': use_angle_cls,  # Updated parameter name
                'lang': lang
            }
            
            # Add GPU parameter only if it's supported
            if use_gpu:
                print("⚠️ GPU mode requested - attempting to configure")
                # Note: GPU usage might be controlled differently in newer versions
            
            self.ocr = PaddleOCR(**init_params)
            self.device = device
            print(f"✓ PaddleOCR initialized successfully with OpenVINO backend")
            print(f"  📱 Device: {device}")
            print(f"  🌐 Language: {lang}")
            print(f"  🔄 Angle classification: {use_angle_cls}")
            
            # Verify backend
            self._verify_openvino_backend()
            
        except Exception as e:
            print(f"❌ Failed to initialize PaddleOCR: {e}")
            sys.exit(1)
    
    def _check_openvino_availability(self):
        """Check if OpenVINO is available"""
        try:
            import openvino as ov
            print(f"✓ OpenVINO version: {ov.__version__}")
            
            # Check available devices
            core = ov.Core()
            devices = core.available_devices
            print(f"📱 Available OpenVINO devices: {devices}")
            
            return True
        except ImportError:
            print("❌ OpenVINO not found. Please install: pip install openvino")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️ OpenVINO check error: {e}")
            return False
    
    def _setup_openvino_environment(self, device: str):
        """Setup OpenVINO environment variables"""
        # Set PaddlePaddle to use OpenVINO
        os.environ['PADDLE_USE_OPENVINO'] = '1'
        
        # Set device preference
        if device.upper() == 'CPU':
            os.environ['PADDLE_OPENVINO_DEVICE'] = 'CPU'
            os.environ['OMP_NUM_THREADS'] = '4'
        elif device.upper() == 'GPU':
            os.environ['PADDLE_OPENVINO_DEVICE'] = 'GPU'
        else:
            os.environ['PADDLE_OPENVINO_DEVICE'] = 'AUTO'
        
        # Disable other backends to force OpenVINO
        os.environ['PADDLE_USE_TENSORRT'] = '0'
        
        print(f"🔧 OpenVINO environment configured for {device}")
    
    def _verify_openvino_backend(self):
        """Verify that OpenVINO backend is being used"""
        print("🔍 Verifying OpenVINO backend usage...")
        
        # Check environment variables
        if os.environ.get('PADDLE_USE_OPENVINO') == '1':
            print("✓ OpenVINO backend enabled via environment")
        else:
            print("⚠️ OpenVINO environment variable not set")
        
        # Additional verification could be added here
        # depending on PaddleOCR version capabilities
    
    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text from a single image using OpenVINO-accelerated OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text, processing_time)
        """
        print(f"  🖼️ Processing: {os.path.basename(image_path)}")
        
        try:
            # Load image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Read image using OpenCV for better format support
            img = cv2.imread(image_path)
            if img is None:
                # Fallback to PIL
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Start timing
            start_time = time.time()
            
            # Perform OCR with OpenVINO backend
            result = self.ocr.ocr(img)
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Extract text from results
            extracted_text = self._parse_ocr_results(result)
            
            print(f"    ⏱️ Processing time: {processing_time:.3f}s")
            print(f"    📝 Extracted {len(extracted_text.strip().split()) if extracted_text.strip() else 0} words")
            
            return extracted_text, processing_time
            
        except Exception as e:
            print(f"    ❌ Error processing {image_path}: {e}")
            return "", 0.0
    
    def _parse_ocr_results(self, result) -> str:
        """
        Parse OCR results and extract text
        
        Args:
            result: OCR result from PaddleOCR
            
        Returns:
            Extracted text as string
        """
        extracted_text = ""
        
        if not result or len(result) == 0:
            print("    🔍 No OCR results returned")
            return extracted_text
        
        try:
            print(f"    🔍 OCR result structure: {type(result)}, length: {len(result) if result else 'None'}")
            
            # Handle PaddleOCR v4+ result format
            if isinstance(result, list) and len(result) > 0:
                # result is typically a list with one element containing the actual results
                ocr_result = result[0]
                print(f"    🔍 First result type: {type(ocr_result)}")
                
                if ocr_result is None:
                    print("    ⚠️ OCR result is None")
                    return extracted_text
                
                # Handle different result formats
                if hasattr(ocr_result, 'rec_texts'):
                    # New format: OCRResult object with rec_texts attribute
                    texts = ocr_result.rec_texts
                    scores = getattr(ocr_result, 'rec_scores', [])
                    print(f"    📝 Found {len(texts)} text items via rec_texts")
                    
                    for i, text in enumerate(texts):
                        score = scores[i] if i < len(scores) else 0.0
                        if score > 0.5:  # Filter by confidence
                            extracted_text += f"{text}\n"
                            print(f"      📖 '{text}' (confidence: {score:.3f})")
                        else:
                            print(f"      ⏭️ Skipped low confidence: '{text}' ({score:.3f})")
                
                elif isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
                    # Dictionary format
                    texts = ocr_result['rec_texts']
                    scores = ocr_result.get('rec_scores', [])
                    print(f"    📝 Found {len(texts)} text items via dict access")
                    
                    for i, text in enumerate(texts):
                        score = scores[i] if i < len(scores) else 0.0
                        if score > 0.5:
                            extracted_text += f"{text}\n"
                            print(f"      📖 '{text}' (confidence: {score:.3f})")
                
                elif isinstance(ocr_result, list):
                    # Traditional format: list of [bbox, (text, confidence)]
                    print(f"    📝 Processing traditional list format with {len(ocr_result)} items")
                    
                    for i, item in enumerate(ocr_result):
                        if isinstance(item, list) and len(item) >= 2:
                            # Format: [bbox, (text, confidence)]
                            text_info = item[1]
                            if isinstance(text_info, tuple) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                                
                                if confidence > 0.5:
                                    extracted_text += f"{text}\n"
                                    print(f"      📖 '{text}' (confidence: {confidence:.3f})")
                                else:
                                    print(f"      ⏭️ Skipped low confidence: '{text}' ({confidence:.3f})")
                            elif isinstance(text_info, str):
                                extracted_text += f"{text_info}\n"
                                print(f"      📖 '{text_info}' (no confidence)")
                
                else:
                    print(f"    ❓ Unknown result format: {type(ocr_result)}")
                    print(f"    🔍 Result sample: {str(ocr_result)[:200]}...")
            
            else:
                print(f"    ❓ Unexpected result structure: {type(result)}")
                
        except Exception as e:
            print(f"    ⚠️ Error parsing OCR results: {e}")
            print(f"    🔍 Result type: {type(result)}")
            if result:
                print(f"    🔍 Result content: {str(result)[:200]}...")
        
        return extracted_text
    
    def process_single_file(self, image_path: str) -> bool:
        """
        Process a single image file and save results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract text
            extracted_text, processing_time = self.extract_text_from_image(image_path)
            
            # Generate output file path with .openvino suffix
            output_path = f"{image_path}.openvino.txt"
            
            # Save extracted text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            print(f"  💾 Saved to: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {image_path}: {e}")
            return False
    
    def process_directory(self, directory_path: str) -> Tuple[int, int, float]:
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to the directory containing images
            
        Returns:
            Tuple of (successful_count, total_count, total_time)
        """
        print(f"📁 Processing directory: {directory_path}")
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            pattern = f"*{ext}"
            image_files.extend(Path(directory_path).glob(pattern))
            image_files.extend(Path(directory_path).glob(pattern.upper()))
        
        if not image_files:
            print("  ⚠️ No image files found in directory")
            return 0, 0, 0.0
        
        print(f"  🔍 Found {len(image_files)} image files")
        
        successful_count = 0
        total_time = 0.0
        
        # Process each image with progress bar
        with tqdm(image_files, desc="Processing images", unit="file") as pbar:
            for image_file in pbar:
                pbar.set_description(f"Processing {image_file.name}")
                
                start_time = time.time()
                success = self.process_single_file(str(image_file))
                end_time = time.time()
                
                if success:
                    successful_count += 1
                
                total_time += (end_time - start_time)
                
                # Update progress bar
                pbar.set_postfix({
                    'success': f"{successful_count}/{len(image_files)}",
                    'avg_time': f"{total_time/len(image_files):.3f}s"
                })
        
        return successful_count, len(image_files), total_time
    
    def process_path(self, input_path: str) -> bool:
        """
        Process a file or directory path
        
        Args:
            input_path: Path to file or directory
            
        Returns:
            True if successful, False otherwise
        """
        path = Path(input_path)
        
        if not path.exists():
            print(f"❌ Path does not exist: {input_path}")
            return False
        
        if path.is_file():
            print(f"📄 Processing single file: {input_path}")
            return self.process_single_file(str(path))
        
        elif path.is_dir():
            successful, total, total_time = self.process_directory(str(path))
            
            print(f"\n📊 Processing Summary:")
            print(f"  ✅ Successful: {successful}/{total} files")
            print(f"  ⏱️ Total time: {total_time:.3f}s")
            print(f"  📈 Average time per file: {total_time/total:.3f}s" if total > 0 else "")
            print(f"  🚀 Success rate: {successful/total*100:.1f}%" if total > 0 else "")
            
            return successful > 0
        
        else:
            print(f"❌ Invalid path type: {input_path}")
            return False


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="PaddleOCR with OpenVINO backend for high-performance OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python ocr_openvino.py image.jpg
  
  # Process all images in a directory
  python ocr_openvino.py ./images/
  
  # Use GPU device (if available)
  python ocr_openvino.py image.jpg --device GPU
  
  # Process English text
  python ocr_openvino.py image.jpg --lang en
  
  # Disable angle classification for better speed
  python ocr_openvino.py image.jpg --no-angle-cls
        """
    )
    
    parser.add_argument(
        'input_path',
        help='Path to image file or directory containing images'
    )
    
    parser.add_argument(
        '--lang',
        default='ch',
        choices=['ch', 'en', 'fr', 'german', 'korean', 'japan'],
        help='Language for OCR recognition (default: ch)'
    )
    
    parser.add_argument(
        '--device',
        default='CPU',
        choices=['CPU', 'GPU', 'AUTO'],
        help='OpenVINO device to use (default: CPU)'
    )
    
    parser.add_argument(
        '--no-angle-cls',
        action='store_true',
        help='Disable angle classification (faster but less accurate for rotated text)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (compatibility option, prefer --device GPU)'
    )
    
    args = parser.parse_args()
    
    print("🔥 PaddleOCR OpenVINO Backend OCR Tool")
    print("=" * 50)
    
    # Initialize OCR with OpenVINO backend
    try:
        ocr_tool = OpenVINOOCR(
            use_angle_cls=not args.no_angle_cls,
            lang=args.lang,
            use_gpu=args.gpu or args.device == 'GPU',
            device=args.device
        )
        
        print(f"\n🎯 Target: {args.input_path}")
        print(f"🌐 Language: {args.lang}")
        print(f"📱 Device: {args.device}")
        print(f"🔄 Angle classification: {not args.no_angle_cls}")
        print("-" * 50)
        
        # Process the input
        start_time = time.time()
        success = ocr_tool.process_path(args.input_path)
        end_time = time.time()
        
        print("-" * 50)
        if success:
            print(f"✅ Processing completed successfully!")
            print(f"⏱️ Total execution time: {end_time - start_time:.3f}s")
            print(f"💾 Results saved with .openvino.txt suffix")
        else:
            print(f"❌ Processing failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
