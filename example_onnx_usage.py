#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX + OpenVINO OCR Usage Examples

This script demonstrates how to use the ocr_openvino_onnx.py tool
for different scenarios and performance optimization.
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ocr_openvino_onnx import OpenVINOONNXOCR
except ImportError as e:
    print(f"❌ Could not import ONNX OCR: {e}")
    sys.exit(1)


def example_single_image_onnx():
    """Example: Process a single image with ONNX backend"""
    print("🖼️ Example 1: ONNX + OpenVINO Single Image Processing")
    print("-" * 50)
    
    try:
        # Initialize ONNX OCR tool
        ocr_tool = OpenVINOONNXOCR(
            lang='ch',
            use_angle_cls=True,
            device='CPU'
        )
        
        # Process test image if it exists
        test_image = "test/pic01.jpg"
        if os.path.exists(test_image):
            success = ocr_tool.process_single_file(test_image)
            if success:
                print(f"✅ Successfully processed {test_image}")
                output_file = f"{test_image}.onnx.txt"
                if os.path.exists(output_file):
                    print(f"📄 Output saved to: {output_file}")
                    
                    # Show first few lines of output
                    with open(output_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:5]
                        print("📝 Sample output:")
                        for line in lines:
                            if line.strip() and not line.startswith('#'):
                                print(f"    {line.strip()}")
            else:
                print(f"❌ Failed to process {test_image}")
        else:
            print(f"⚠️ Test image not found: {test_image}")
            
    except Exception as e:
        print(f"❌ Error in example: {e}")


def example_performance_comparison():
    """Example: Compare different OCR backends"""
    print("\n⚡ Example 2: Performance Comparison")
    print("-" * 50)
    
    test_image = "test/pic01.jpg"
    if not os.path.exists(test_image):
        print(f"⚠️ Test image not found: {test_image}")
        return
    
    backends = [
        {"name": "ONNX + OpenVINO (with angle cls)", "use_angle_cls": True},
        {"name": "ONNX + OpenVINO (without angle cls)", "use_angle_cls": False}
    ]
    
    for backend in backends:
        print(f"\n🧪 Testing: {backend['name']}")
        
        try:
            ocr_tool = OpenVINOONNXOCR(
                lang='ch',
                use_angle_cls=backend['use_angle_cls'],
                device='CPU'
            )
            
            # Multiple runs for better timing
            times = []
            for i in range(3):
                start_time = time.time()
                text, processing_time = ocr_tool.extract_text_from_image(test_image)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            print(f"  ⏱️ Average processing time: {avg_time:.3f}s")
            print(f"  📊 Times: {[f'{t:.3f}s' for t in times]}")
            
        except Exception as e:
            print(f"  ❌ Error testing {backend['name']}: {e}")


def example_batch_processing():
    """Example: Batch process directory with ONNX backend"""
    print("\n📁 Example 3: Batch Processing with ONNX")
    print("-" * 50)
    
    try:
        # Initialize ONNX OCR tool for fast batch processing
        ocr_tool = OpenVINOONNXOCR(
            lang='ch',
            use_angle_cls=False,  # Disable for faster processing
            device='CPU'
        )
        
        # Process test directory if it exists
        test_dir = "test"
        if os.path.exists(test_dir):
            print(f"🚀 Processing directory: {test_dir}")
            start_time = time.time()
            
            successful, total, total_time = ocr_tool.process_directory(test_dir)
            
            end_time = time.time()
            
            print(f"\n📊 Batch Processing Results:")
            print(f"  ✅ Successful: {successful}/{total} files")
            print(f"  ⏱️ Total processing time: {total_time:.3f}s")
            print(f"  📈 Average time per file: {total_time/total:.3f}s" if total > 0 else "")
            print(f"  🚀 Success rate: {successful/total*100:.1f}%" if total > 0 else "")
            print(f"  ⚡ Overall execution time: {end_time - start_time:.3f}s")
            
        else:
            print(f"⚠️ Test directory not found: {test_dir}")
            
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")


def example_different_languages():
    """Example: Process text in different languages"""
    print("\n🌐 Example 4: Multi-language Processing")
    print("-" * 50)
    
    languages = [
        {'lang': 'ch', 'name': 'Chinese'},
        {'lang': 'en', 'name': 'English'}
    ]
    
    test_image = "test/pic01.jpg"
    if not os.path.exists(test_image):
        print(f"⚠️ Test image not found: {test_image}")
        return
    
    for lang_config in languages:
        print(f"\n🔤 Testing {lang_config['name']} OCR:")
        
        try:
            ocr_tool = OpenVINOONNXOCR(
                lang=lang_config['lang'],
                use_angle_cls=True,
                device='CPU'
            )
            
            text, processing_time = ocr_tool.extract_text_from_image(test_image)
            
            print(f"  ⏱️ Processing time: {processing_time:.3f}s")
            print(f"  📝 Text length: {len(text)} characters")
            if text.strip():
                # Show first line of extracted text
                first_line = text.split('\n')[0]
                print(f"  📖 Sample: '{first_line}'")
            
        except Exception as e:
            print(f"  ❌ Error processing {lang_config['name']}: {e}")


def show_model_info():
    """Show information about ONNX models"""
    print("\n🔧 Example 5: ONNX Model Information")
    print("-" * 50)
    
    model_dir = "onnx_models"
    if os.path.exists(model_dir):
        print(f"📁 ONNX models directory: {model_dir}")
        
        model_files = list(Path(model_dir).glob("*.onnx"))
        if model_files:
            print(f"📋 Found {len(model_files)} ONNX models:")
            for model_file in model_files:
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"  📄 {model_file.name} ({size_mb:.2f} MB)")
        else:
            print("  ⚠️ No ONNX model files found")
    else:
        print(f"⚠️ ONNX models directory not found: {model_dir}")
    
    print("\n💡 To convert models manually:")
    print("  python ocr_openvino_onnx.py --convert-only --lang ch")


if __name__ == "__main__":
    print("🔥 ONNX + OpenVINO OCR Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_single_image_onnx()
        example_performance_comparison()
        example_batch_processing()
        example_different_languages()
        show_model_info()
        
        print("\n" + "=" * 60)
        print("✅ All ONNX examples completed!")
        print("\n💡 Key advantages of ONNX + OpenVINO backend:")
        print("  🚀 Maximum performance optimization")
        print("  📦 Portable model format (ONNX)")
        print("  ⚡ Hardware-accelerated inference")
        print("  🔧 Cross-platform deployment")
        print("  📈 Scalable for production use")
        print("\n🔗 Usage:")
        print("  python ocr_openvino_onnx.py image.jpg --device CPU")
        print("  python ocr_openvino_onnx.py ./images/ --no-angle-cls")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        sys.exit(1)
