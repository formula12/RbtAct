#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例运行脚本
演示如何使用数据集生成工具
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """检查运行环境和文件"""
    print("=== 检查运行环境 ===")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要文件
    required_files = [
        "generate_sft_dataset.py",
        "config.py"
    ]
    
    # 动态检查配置文件中指定的输入文件
    try:
        from config import JSONL_FILE
        required_files.append(JSONL_FILE)
    except ImportError:
        required_files.append("iclr2024_map_merged_per_filter2.jsonl")  # 默认文件
    
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
            # 检查目录中的文件数量
            md_files = list(Path(dir_path).glob("*.md"))
            print(f"✅ {dir_path} (包含 {len(md_files)} 个.md文件)")
    
    if missing_files:
        print(f"\n❌ 缺少文件: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"\n❌ 缺少目录: {missing_dirs}")
        return False
    
    print("\n✅ 所有必要文件和目录都存在")
    return True

def run_generation():
    """运行数据集生成"""
    print("\n=== 开始生成数据集 ===")
    
    try:
        # 导入并运行主脚本
        from generate_sft_dataset import main
        main()
        return True
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return False

def run_test():
    """运行测试脚本"""
    print("\n=== 测试生成的数据集 ===")
    
    try:
        from test_dataset import test_dataset_format, analyze_perspectives
        
        # 从配置文件获取输出文件名
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
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("ShareGPT格式SFT数据集生成工具")
    print("=" * 50)
    
    # 检查环境
    if not check_requirements():
        print("\n请确保所有必要文件都存在后再运行")
        return
    
    # 生成数据集
    if not run_generation():
        print("\n数据集生成失败，请检查错误信息")
        return
    
    # 测试数据集
    if not run_test():
        print("\n数据集测试失败")
        return
    
    print("\n🎉 数据集生成和验证完成!")
    print("\n下一步:")
    print("1. 检查生成的 sft_dataset_sharegpt.json 文件")
    print("2. 根据需要调整 config.py 中的参数")
    print("3. 将数据集用于llama-factory训练")

if __name__ == "__main__":
    main()

