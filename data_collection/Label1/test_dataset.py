#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试生成的数据集格式
"""

import json
from pathlib import Path

def test_dataset_format(dataset_file: str):
    """测试数据集格式是否正确"""
    
    if not Path(dataset_file).exists():
        print(f"数据集文件 {dataset_file} 不存在")
        return False
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return False
    
    if not isinstance(data, list):
        print("数据集应该是一个列表")
        return False
    
    print(f"数据集包含 {len(data)} 条样本")
    
    # 检查前几个样本的格式
    for i, sample in enumerate(data[:3]):
        print(f"\n=== 样本 {i+1} ===")
        
        if not isinstance(sample, dict):
            print(f"样本 {i+1} 不是字典格式")
            return False
        
        # 检查必要字段
        required_fields = ["conversations", "conference", "paper_id", "paper_title", "review_id", "perspective"]
        for field in required_fields:
            if field not in sample:
                print(f"样本 {i+1} 缺少 {field} 字段")
                return False
        
        conversations = sample["conversations"]
        if not isinstance(conversations, list) or len(conversations) != 3:
            print(f"样本 {i+1} conversations 应该包含3条消息")
            return False
        
        # 检查角色
        expected_roles = ["system", "human", "gpt"]
        for j, msg in enumerate(conversations):
            if msg.get("from") != expected_roles[j]:
                print(f"样本 {i+1} 消息 {j+1} 角色错误，期望 {expected_roles[j]}")
                return False
            
            if not msg.get("value"):
                print(f"样本 {i+1} 消息 {j+1} 内容为空")
                return False
        
        # 打印样本元数据
        print(f"Conference: {sample['conference']}")
        print(f"Paper ID: {sample['paper_id']}")
        print(f"Paper Title: {sample['paper_title'][:50]}..." if len(sample['paper_title']) > 50 else f"Paper Title: {sample['paper_title']}")
        print(f"Review ID: {sample['review_id']}")
        print(f"Perspective: {sample['perspective']}")
        
        # 打印样本内容（截断显示）
        print(f"System: {conversations[0]['value']}")
        user_content = conversations[1]['value']
        if len(user_content) > 200:
            user_content = user_content[:200] + "..."
        print(f"Human: {user_content}")
        
        assistant_content = conversations[2]['value']
        if len(assistant_content) > 200:
            assistant_content = assistant_content[:200] + "..."
        print(f"GPT: {assistant_content}")
    
    print(f"\n✅ 数据集格式验证通过!")
    return True

def analyze_perspectives(dataset_file: str):
    """分析数据集中各perspective的分布"""
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    perspective_count = {}
    conference_count = {}
    
    for sample in data:
        # 直接从perspective字段读取
        perspective = sample.get("perspective", "Unknown")
        perspective_count[perspective] = perspective_count.get(perspective, 0) + 1
        
        # 统计会议分布
        conference = sample.get("conference", "Unknown")
        conference_count[conference] = conference_count.get(conference, 0) + 1
    
    print(f"\n=== Perspective分布 ===")
    for perspective, count in sorted(perspective_count.items()):
        print(f"{perspective}: {count}条")
    
    print(f"\n=== Conference分布 ===")
    for conference, count in sorted(conference_count.items()):
        print(f"{conference}: {count}条")
    
    total = sum(perspective_count.values())
    print(f"\n总计: {total}条")

if __name__ == "__main__":
    dataset_file = "sft_dataset_sharegpt.json"
    
    if test_dataset_format(dataset_file):
        analyze_perspectives(dataset_file)
    else:
        print("数据集格式验证失败")
