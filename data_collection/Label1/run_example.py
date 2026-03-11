#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example runner script
Demonstrates how to use the dataset generation tool
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check the environment and required files"""
    print("=== Checking environment ===")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required files
    required_files = [
        "generate_sft_dataset.py",
        "config.py"
    ]
    
    # Dynamically check the input file specified in config
    try:  
        from config import JSONL_FILE
        required_files.append(JSONL_FILE)
    except ImportError:
        required_files.append("iclr2024_map_merged_per_filter2.jsonl")  # Default file
    
    required_dirs = [
        "paper_md/iclr2024"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            # Check the number of files in the directory
            md_files = list(Path(dir_path).glob("*.md"))
            print(f"✅ {dir_path} ({len(md_files)} .md files)")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False
    
    print("\n✅ All required files and directories are available")
    return True

def run_generation():
    """Run dataset generation"""
    print("\n=== Generating dataset ===")
    
    try:
        # Import and run the main script
        from generate_sft_dataset import main
        main()
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def run_test():
    """Run the test script"""
    print("\n=== Testing generated dataset ===")
    
    try:
        from test_dataset import test_dataset_format, analyze_perspectives
        
        # Get the output filename from config
        try:
            from config import OUTPUT_FILE
            dataset_file = OUTPUT_FILE
        except ImportError:
            dataset_file = "sft_dataset_sharegpt.json"
        if test_dataset_format(dataset_file):
            analyze_perspectives(dataset_file)
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("ShareGPT SFT Dataset Generator")
    print("=" * 50)
    
    # Check environment
    if not check_requirements():
        print("\nPlease make sure all required files are available before running")
        return
    
    # Generate dataset
    if not run_generation():
        print("\nDataset generation failed. Please check the error message")
        return
    
    # Test dataset
    if not run_test():
        print("\nDataset test failed")
        return
    
    print("\n🎉 Dataset generation and validation completed!")
    print("\nNext steps:")
    print("1. Check the generated sft_dataset_sharegpt.json file")
    print("2. Adjust parameters in config.py if needed")
    print("3. Use the dataset for LLaMA-Factory training")

if __name__ == "__main__":
    main()