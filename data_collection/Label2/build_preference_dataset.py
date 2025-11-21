#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构造DPO训练的偏好数据对
从weakness-rebuttal映射数据中生成chosen/rejected对
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

# 配置参数
class Config:
    # 输入输出路径
    INPUT_JSONL = "iclr2024_map_merged_per_filter2_imp_train.jsonl"
    PAPER_MD_DIR = "paper_md/iclr2024"
    OUTPUT_JSON = "preference_dataset_dpo.json"
    
    # 采样参数
    TARGET_TOTAL_PAIRS = 10000  # 目标总对数
    TARGET_PAIRS_PER_PERSPECTIVE = 1500  # 每个perspective的目标对数
    MIN_PERSPECTIVE_RATIO = 0.10  # 每个perspective最小占比 10%
    MAX_SAMPLE_REUSE = 3  # 每个样本最大复用次数
    
    # perspective列表
    PERSPECTIVES = ["Evaluation", "Experiments", "Novelty", "Presentation", 
                   "Reproducibility", "Theory", "Writing"]
    
    # impact排序 (好→差)
    IMPACT_ORDER = {"CRP": 5, "SRP": 4, "VCR": 3, "DWC": 2, "DRF": 1}
    
    # 分层配对规则 (优先级从高到低)
    TIER_RULES = [
        # Tier-1 (Easy - 大间隔)
        ("CRP", ["DWC", "DRF"], "easy", 0.4),  # 40%
        # Tier-2 (Medium - 中等间隔)
        ("SRP", ["DWC", "DRF"], "medium", 0.25),  # 25%
        ("CRP", ["VCR"], "medium", 0.15),  # 15%
        # Tier-3 (Hard - 近邻间隔)
        ("CRP", ["SRP"], "hard", 0.10),  # 10%
        ("VCR", ["DWC", "DRF"], "hard", 0.08),  # 8%
        ("DWC", ["DRF"], "hard", 0.02),  # 2% (兜底)
    ]


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_paper_content(paper_id: str, paper_md_dir: str) -> str:
    """加载论文内容"""
    paper_path = Path(paper_md_dir) / f"{paper_id}.md"
    if paper_path.exists():
        with open(paper_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def group_by_paper_and_perspective(data: List[Dict]) -> Dict:
    """
    按论文和perspective分组weakness points
    返回: {paper_id: {perspective: [weakness_items]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for paper in data:
        paper_id = paper["paper_id"]
        paper_title = paper["paper_title"]
        
        for mapping in paper["weakness_rebuttal_mappings"]:
            perspective = mapping["weakness_point"]["perspective"]
            
            # 构造完整的weakness item
            item = {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "review_id": mapping["review_id"],
                "weakness_id": mapping["weakness_point"]["id"],
                "weakness_content": mapping["weakness_point"]["content"],
                "perspective": perspective,
                "rebuttal_id": mapping["rebuttal_response"]["id"],
                "rebuttal_content": mapping["rebuttal_response"]["content"],
                "impact": mapping["rebuttal_response"]["impact"],
                "confidence_score": mapping.get("confidence_score", 0.0)
            }
            
            grouped[paper_id][perspective].append(item)
    
    return grouped


def can_pair(chosen_impact: str, rejected_impact: str, tier_rule: Tuple) -> bool:
    """判断两个impact是否可以按照给定规则配对"""
    chosen_label, rejected_labels, _, _ = tier_rule
    
    if chosen_impact == chosen_label and rejected_impact in rejected_labels:
        # 确保chosen确实优于rejected
        if Config.IMPACT_ORDER[chosen_impact] > Config.IMPACT_ORDER[rejected_impact]:
            return True
    
    return False


def create_preference_pair(chosen_item: Dict, rejected_item: Dict, 
                          paper_content: str) -> Dict:
    """创建一个偏好数据对"""
    instruction = "You are a professional reviewer. Provide a constructive comment on the given scientific paper."
    
    input_text = f"""[Request]: From the perspective of {chosen_item['perspective'][0].lower() + chosen_item['perspective'][1:]}, provide a constructive comment on the above paper.

[BEGIN PAPER]
{paper_content}
[END PAPER]"""
    
    pair = {
        "instruction": instruction,
        "input": input_text,
        "chosen": chosen_item["weakness_content"],
        "rejected": rejected_item["weakness_content"],
        "conference": "ICLR-2024",
        "paper_id": chosen_item["paper_id"],
        "paper_title": chosen_item["paper_title"],
        "review_id": {
            "chosen": chosen_item["review_id"],
            "rejected": rejected_item["review_id"]
        },
        "perspective": chosen_item["perspective"],
        "rebuttal_labels": {
            "chosen": chosen_item["impact"],
            "rejected": rejected_item["impact"]
        }
    }
    
    return pair


def sample_pairs_for_perspective(grouped_data: Dict, perspective: str, 
                                 target_count: int, paper_md_dir: str,
                                 paper_usage: Dict, sample_usage: Dict) -> List[Dict]:
    """
    为特定perspective采样配对
    
    Args:
        grouped_data: 分组后的数据
        perspective: 要采样的perspective
        target_count: 目标对数
        paper_md_dir: 论文markdown目录
        paper_usage: 论文使用次数统计
        sample_usage: 样本使用次数统计
    """
    pairs = []
    
    # 收集该perspective下所有可配对的候选
    candidates = []
    for paper_id, perspectives_dict in grouped_data.items():
        if perspective not in perspectives_dict:
            continue
        
        items = perspectives_dict[perspective]
        if len(items) < 2:  # 至少需要2个才能配对
            continue
        
        candidates.append((paper_id, items))
    
    if not candidates:
        return pairs
    
    # 按论文使用次数排序，优先使用次数少的论文
    candidates.sort(key=lambda x: paper_usage.get(x[0], 0))
    
    # 按tier规则分层采样
    tier_allocations = []
    for tier_rule in Config.TIER_RULES:
        tier_target = int(target_count * tier_rule[3])
        tier_allocations.append((tier_rule, tier_target))
    
    # 对每个tier进行采样
    for tier_rule, tier_target in tier_allocations:
        tier_pairs = []
        
        # 随机打乱candidates以增加多样性
        random.shuffle(candidates)
        
        for paper_id, items in candidates:
            if len(tier_pairs) >= tier_target:
                break
            
            # 加载论文内容
            paper_content = load_paper_content(paper_id, paper_md_dir)
            if paper_content is None:
                continue
            
            # 尝试在该论文中找配对
            for i, item_i in enumerate(items):
                for j, item_j in enumerate(items):
                    if i == j:
                        continue
                    
                    # 检查是否符合tier规则
                    if can_pair(item_i["impact"], item_j["impact"], tier_rule):
                        # 检查样本复用次数
                        sample_key_i = f"{item_i['paper_id']}_{item_i['review_id']}_{item_i['weakness_id']}"
                        sample_key_j = f"{item_j['paper_id']}_{item_j['review_id']}_{item_j['weakness_id']}"
                        
                        if (sample_usage.get(sample_key_i, 0) < Config.MAX_SAMPLE_REUSE and
                            sample_usage.get(sample_key_j, 0) < Config.MAX_SAMPLE_REUSE):
                            
                            # 创建配对
                            pair = create_preference_pair(item_i, item_j, paper_content)
                            tier_pairs.append(pair)
                            
                            # 更新使用计数
                            paper_usage[paper_id] = paper_usage.get(paper_id, 0) + 1
                            sample_usage[sample_key_i] = sample_usage.get(sample_key_i, 0) + 1
                            sample_usage[sample_key_j] = sample_usage.get(sample_key_j, 0) + 1
                            
                            if len(tier_pairs) >= tier_target:
                                break
                
                if len(tier_pairs) >= tier_target:
                    break
        
        pairs.extend(tier_pairs)
        print(f"  {tier_rule[2].capitalize()} tier ({tier_rule[0]} × {tier_rule[1]}): 采样了 {len(tier_pairs)} 对")
    
    return pairs


def main():
    print("=" * 80)
    print("开始构造DPO偏好数据集")
    print("=" * 80)
    
    # 加载数据
    print(f"\n[加载数据] {Config.INPUT_JSONL}")
    data = load_jsonl(Config.INPUT_JSONL)
    print(f"   加载了 {len(data)} 篇论文")
    
    # 按论文和perspective分组
    print("\n[分组数据] 按论文和perspective分组...")
    grouped_data = group_by_paper_and_perspective(data)
    print(f"   共 {len(grouped_data)} 篇论文")
    
    # 统计每个perspective的可用数据
    print("\n[数据统计] 各perspective的数据分布:")
    perspective_stats = defaultdict(int)
    for paper_id, perspectives_dict in grouped_data.items():
        for perspective, items in perspectives_dict.items():
            perspective_stats[perspective] += len(items)
    
    for perspective in Config.PERSPECTIVES:
        count = perspective_stats.get(perspective, 0)
        print(f"   {perspective:15s}: {count:5d} 个weakness points")
    
    # 检查缺失的论文
    print(f"\n[检查论文] 检查论文文件...")
    missing_papers = []
    available_papers = []
    for paper_id in grouped_data.keys():
        if load_paper_content(paper_id, Config.PAPER_MD_DIR) is None:
            missing_papers.append(paper_id)
        else:
            available_papers.append(paper_id)
    
    print(f"   可用论文: {len(available_papers)}")
    print(f"   缺失论文: {len(missing_papers)}")
    if missing_papers:
        print(f"\n[警告] 缺失的论文文件 (前20个):")
        for paper_id in missing_papers[:20]:
            print(f"      - {paper_id}.md")
        if len(missing_papers) > 20:
            print(f"      ... 还有 {len(missing_papers) - 20} 个")
    
    # 采样偏好数据对
    print(f"\n[开始采样] 开始采样偏好数据对...")
    print(f"   目标总数: {Config.TARGET_TOTAL_PAIRS}")
    print(f"   每个perspective目标: {Config.TARGET_PAIRS_PER_PERSPECTIVE}")
    print(f"   最大样本复用次数: {Config.MAX_SAMPLE_REUSE}")
    
    all_pairs = []
    paper_usage = {}  # 记录每篇论文使用次数
    sample_usage = {}  # 记录每个样本使用次数
    
    for perspective in Config.PERSPECTIVES:
        print(f"\n[处理] {perspective}:")
        pairs = sample_pairs_for_perspective(
            grouped_data, perspective, Config.TARGET_PAIRS_PER_PERSPECTIVE,
            Config.PAPER_MD_DIR, paper_usage, sample_usage
        )
        print(f"   总共采样: {len(pairs)} 对")
        all_pairs.extend(pairs)
    
    # 打乱顺序
    random.shuffle(all_pairs)
    
    # 保存结果
    print(f"\n[保存结果] 保存到: {Config.OUTPUT_JSON}")
    with open(Config.OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    
    # 统计报告
    print("\n" + "=" * 80)
    print("生成数据集统计报告")
    print("=" * 80)
    
    print(f"\n[完成] 总共生成: {len(all_pairs)} 对偏好数据")
    
    # 按perspective统计
    perspective_counts = defaultdict(int)
    for pair in all_pairs:
        perspective_counts[pair["perspective"]] += 1
    
    print(f"\n[统计1] 各Perspective分布:")
    for perspective in Config.PERSPECTIVES:
        count = perspective_counts.get(perspective, 0)
        ratio = count / len(all_pairs) * 100 if all_pairs else 0
        print(f"   {perspective:15s}: {count:5d} 对 ({ratio:5.2f}%)")
    
    # 按难度层级统计
    tier_counts = defaultdict(int)
    for pair in all_pairs:
        chosen_impact = pair["rebuttal_labels"]["chosen"]
        rejected_impact = pair["rebuttal_labels"]["rejected"]
        pair_type = f"{chosen_impact} x {rejected_impact}"
        tier_counts[pair_type] += 1
    
    print(f"\n[统计2] 配对类型分布:")
    for pair_type, count in sorted(tier_counts.items(), key=lambda x: -x[1]):
        ratio = count / len(all_pairs) * 100 if all_pairs else 0
        print(f"   {pair_type:15s}: {count:5d} 对 ({ratio:5.2f}%)")
    
    # 论文使用分布
    print(f"\n[统计3] 论文使用统计:")
    print(f"   使用的论文总数: {len(paper_usage)}")
    usage_distribution = defaultdict(int)
    for count in paper_usage.values():
        usage_distribution[count] += 1
    
    for usage_count in sorted(usage_distribution.keys()):
        paper_count = usage_distribution[usage_count]
        print(f"   使用 {usage_count} 次的论文: {paper_count} 篇")
    
    # 样本复用统计
    max_reuse = max(sample_usage.values()) if sample_usage else 0
    print(f"\n[统计4] 样本复用统计:")
    print(f"   最大复用次数: {max_reuse}")
    
    reuse_distribution = defaultdict(int)
    for count in sample_usage.values():
        reuse_distribution[count] += 1
    
    for reuse_count in sorted(reuse_distribution.keys()):
        sample_count = reuse_distribution[reuse_count]
        print(f"   复用 {reuse_count} 次的样本: {sample_count} 个")
    
    print("\n[完成] 数据集构建完成!")
    print("=" * 80)


if __name__ == "__main__":
    random.seed(42)  # 设置随机种子以保证可复现
    main()
