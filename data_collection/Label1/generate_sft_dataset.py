#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成ShareGPT格式的SFT数据集
用于segment级别的peer review generation任务
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# 导入配置
from config import (
    JSONL_FILE, PAPER_MD_DIR, OUTPUT_FILE, 
    SAMPLES_PER_PERSPECTIVE, MIN_CONFIDENCE, RANDOM_SEED,
    PERSPECTIVES, PERSPECTIVE_MAP, SYSTEM_PROMPT, USER_REQUEST_TEMPLATE
)

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_paper_content(paper_id: str, paper_md_dir: str) -> Optional[str]:
    """加载论文内容"""
    paper_path = Path(paper_md_dir) / f"{paper_id}.md"
    if not paper_path.exists():
        return None
    
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取论文 {paper_id} 时出错: {e}")
        return None

def create_sharegpt_sample(
    paper_content: str, 
    perspective: str, 
    weakness_content: str,
    paper_id: str,
    paper_title: str,
    review_id: str,
    conference: str = "ICLR2024"
) -> Dict:
    """创建ShareGPT格式的样本"""
    
    perspective_text = PERSPECTIVE_MAP.get(perspective, perspective.lower())
    user_content = USER_REQUEST_TEMPLATE.format(
        paper_content=paper_content,
        perspective_text=perspective_text
    )

    return {
        "conversations": [
            {
                "from": "system",
                "value": SYSTEM_PROMPT
            },
            {
                "from": "human", 
                "value": user_content
            },
            {
                "from": "gpt",
                "value": weakness_content
            }
        ],
        "conference": conference,
        "paper_id": paper_id,
        "paper_title": paper_title,
        "review_id": review_id,
        "perspective": perspective
    }

def sample_data_by_perspective(
    data: List[Dict], 
    paper_md_dir: str,
    samples_per_perspective: int = 300,
    min_confidence: float = 0.9
) -> List[Dict]:
    """
    按perspective采样数据
    
    Args:
        data: 原始数据列表
        paper_md_dir: 论文markdown文件目录
        samples_per_perspective: 每个perspective采样数量
        min_confidence: 最小置信度阈值
    
    Returns:
        采样后的ShareGPT格式数据列表
    """
    
    # 使用配置文件中定义的perspectives
    perspectives = PERSPECTIVES
    
    # 按perspective分组收集数据
    perspective_data = defaultdict(list)
    missing_papers = set()
    
    print("正在收集数据...")
    for paper in data:
        paper_id = paper["paper_id"]
        
        # 检查论文文件是否存在
        paper_content = load_paper_content(paper_id, paper_md_dir)
        if paper_content is None:
            missing_papers.add(paper_id)
            continue
            
        # 收集该论文的所有weakness_point
        for mapping in paper.get("weakness_rebuttal_mappings", []):
            weakness_point = mapping.get("weakness_point", {})
            confidence_score = mapping.get("confidence_score", 0.0)
            review_id = mapping.get("review_id", "")
            
            perspective = weakness_point.get("perspective")
            content = weakness_point.get("content")
            
            if perspective in perspectives and content and confidence_score >= min_confidence:
                perspective_data[perspective].append({
                    "paper_id": paper_id,
                    "paper_title": paper.get("paper_title", ""),
                    "conference": paper.get("conference", "ICLR-2024"),
                    "paper_content": paper_content,
                    "weakness_content": content,
                    "confidence_score": confidence_score,
                    "review_id": review_id,
                    "perspective": perspective
                })
    
    if missing_papers:
        print(f"找不到以下论文文件: {sorted(list(missing_papers))}")
        print(f"缺失论文数量: {len(missing_papers)}")
    
    # 统计每个perspective的数据量
    print("\n各perspective数据统计:")
    for perspective in perspectives:
        print(f"{perspective}: {len(perspective_data[perspective])}条")
    
    # 采样数据
    sampled_data = []
    sampling_stats = {}
    
    for perspective in perspectives:
        available_data = perspective_data[perspective]
        
        if not available_data:
            print(f"警告: {perspective} 没有可用数据")
            sampling_stats[perspective] = 0
            continue
        
        # 按论文去重 - 每篇论文最多选一条
        paper_to_samples = defaultdict(list)
        for sample in available_data:
            paper_to_samples[sample["paper_id"]].append(sample)
        
        # 每篇论文选择置信度最高的一条
        unique_samples = []
        for paper_id, samples in paper_to_samples.items():
            best_sample = max(samples, key=lambda x: x["confidence_score"])
            unique_samples.append(best_sample)
        
        # 随机采样
        sample_count = min(samples_per_perspective, len(unique_samples))
        selected_samples = random.sample(unique_samples, sample_count)
        
        # 转换为ShareGPT格式
        for sample in selected_samples:
            sharegpt_sample = create_sharegpt_sample(
                paper_content=sample["paper_content"],
                perspective=sample["perspective"],
                weakness_content=sample["weakness_content"],
                paper_id=sample["paper_id"],
                paper_title=sample["paper_title"],
                review_id=sample["review_id"],
                conference=sample["conference"]
            )
            sampled_data.append(sharegpt_sample)
        
        sampling_stats[perspective] = sample_count
        
        if sample_count < samples_per_perspective:
            print(f"警告: {perspective} 只采集到 {sample_count} 条数据，少于目标 {samples_per_perspective} 条")
    
    print(f"\n采样完成统计:")
    total_samples = 0
    for perspective in perspectives:
        count = sampling_stats.get(perspective, 0)
        print(f"{perspective}: {count}条")
        total_samples += count
    
    print(f"总计: {total_samples}条")
    
    return sampled_data

def main():
    """主函数"""
    print(f"开始生成SFT数据集...")
    print(f"输入文件: {JSONL_FILE}")
    print(f"论文目录: {PAPER_MD_DIR}")
    print(f"每个perspective采样: {SAMPLES_PER_PERSPECTIVE}条")
    print(f"最小置信度: {MIN_CONFIDENCE}")
    print(f"输出文件: {OUTPUT_FILE}")
    
    # 检查输入文件
    if not os.path.exists(JSONL_FILE):
        print(f"错误: 找不到输入文件 {JSONL_FILE}")
        return
    
    if not os.path.exists(PAPER_MD_DIR):
        print(f"错误: 找不到论文目录 {PAPER_MD_DIR}")
        return
    
    # 加载数据
    print("加载数据...")
    data = load_jsonl(JSONL_FILE)
    print(f"加载了 {len(data)} 篇论文的数据")
    
    # 设置随机种子以确保可重现性
    random.seed(RANDOM_SEED)
    
    # 采样数据
    sampled_data = sample_data_by_perspective(
        data, 
        PAPER_MD_DIR, 
        samples_per_perspective=SAMPLES_PER_PERSPECTIVE,
        min_confidence=MIN_CONFIDENCE
    )
    
    # 保存结果
    print(f"\n保存数据到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据集生成完成! 共 {len(sampled_data)} 条样本")
    print(f"文件已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
