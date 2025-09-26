#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of ocr_openvino.py

This script demonstrates how to use the OpenVINO-based OCR tool
for different scenarios.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_openvino import OpenVINOOCR


def example_single_image():
    """Example: Process a single image"""
    print("🖼️ Example 1: Processing a single image")
    print("-" * 40)
    
    # Initialize OCR tool
    ocr_tool = OpenVINOOCR(
        use_angle_cls=True,
        lang='ch',
        device='CPU'
    )
    
    # Process test image if it exists
    test_image = "test/pic01.jpg"
    if os.path.exists(test_image):
        success = ocr_tool.process_single_file(test_image)
        if success:
            print(f"✅ Successfully processed {test_image}")
            output_file = f"{test_image}.openvino.txt"
            if os.path.exists(output_file):
                print(f"📄 Output saved to: {output_file}")
        else:
            print(f"❌ Failed to process {test_image}")
    else:
        print(f"⚠️ Test image not found: {test_image}")


def example_directory():
    """Example: Process all images in a directory"""
    print("\n📁 Example 2: Processing a directory")
    print("-" * 40)
    
    # Initialize OCR tool with different settings
    ocr_tool = OpenVINOOCR(
        use_angle_cls=False,  # Disable for faster processing
        lang='ch',
        device='CPU'
    )
    
    # Process test directory if it exists
    test_dir = "test"
    if os.path.exists(test_dir):
        successful, total, total_time = ocr_tool.process_directory(test_dir)
        print(f"📊 Processed {successful}/{total} files in {total_time:.3f}s")
    else:
        print(f"⚠️ Test directory not found: {test_dir}")


def example_english_text():
    """Example: Process English text with different settings"""
    print("\n🔤 Example 3: Processing English text")
    print("-" * 40)
    
    # Initialize OCR tool for English
    ocr_tool = OpenVINOOCR(
        use_angle_cls=True,
        lang='en',  # English language
        device='CPU'
    )
    
    # You can process English images here
    print("🌐 OCR tool configured for English text processing")
    print("💡 Use this configuration for English documents")


def benchmark_performance():
    """Example: Simple performance benchmark"""
    print("\n⚡ Example 4: Performance benchmark")
    print("-" * 40)
    
    test_image = "test/pic01.jpg"
    if not os.path.exists(test_image):
        print(f"⚠️ Test image not found: {test_image}")
        return
    
    # Test with different configurations
    configs = [
        {"use_angle_cls": True, "name": "With angle classification"},
        {"use_angle_cls": False, "name": "Without angle classification"},
    ]
    
    for config in configs:
        print(f"\n🧪 Testing: {config['name']}")
        
        ocr_tool = OpenVINOOCR(
            use_angle_cls=config['use_angle_cls'],
            lang='ch',
            device='CPU'
        )
        
        # Process and measure time
        text, processing_time = ocr_tool.extract_text_from_image(test_image)
        print(f"  ⏱️ Processing time: {processing_time:.3f}s")
        print(f"  📝 Text length: {len(text)} characters")


if __name__ == "__main__":
    print("🚀 OpenVINO OCR Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_single_image()
        example_directory()
        example_english_text()
        benchmark_performance()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        print("\n💡 Usage tips:")
        print("  - Use --device CPU for most stable performance")
        print("  - Use --device GPU if you have OpenVINO GPU support")
        print("  - Disable angle classification (--no-angle-cls) for faster processing")
        print("  - Results are saved with .openvino.txt suffix")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        sys.exit(1)
