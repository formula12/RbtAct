#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ShareGPT-format SFT dataset
for segment-level peer review generation
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import config
from config import (
    JSONL_FILE, PAPER_MD_DIR, OUTPUT_FILE, 
    SAMPLES_PER_PERSPECTIVE, MIN_CONFIDENCE, RANDOM_SEED,
    PERSPECTIVES, PERSPECTIVE_MAP, SYSTEM_PROMPT, USER_REQUEST_TEMPLATE
)

def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_paper_content(paper_id: str, paper_md_dir: str) -> Optional[str]:
    """Load paper content"""
    paper_path = Path(paper_md_dir) / f"{paper_id}.md"
    if not paper_path.exists():
        return None
    
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
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
    """Create a ShareGPT-format sample"""
    
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
    Sample data by perspective
    
    Args:
        data: Raw data list
        paper_md_dir: Directory of paper markdown files
        samples_per_perspective: Number of samples per perspective
        min_confidence: Minimum confidence threshold
    
    Returns:
        Sampled data list in ShareGPT format
    """
    
    # Use perspectives defined in the config file
    perspectives = PERSPECTIVES
    
    # Group data by perspective
    perspective_data = defaultdict(list)
    missing_papers = set()
    
    print("Collecting data...")
    for paper in data:
        paper_id = paper["paper_id"]
        
        # Check whether the paper file exists
        paper_content = load_paper_content(paper_id, paper_md_dir)
        if paper_content is None:
            missing_papers.add(paper_id)
            continue
            
        # Collect all weakness points for this paper
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
        print(f"Missing paper files: {sorted(list(missing_papers))}")
        print(f"Number of missing papers: {len(missing_papers)}")
    
    # Show data count for each perspective
    print("\nData count by perspective:")
    for perspective in perspectives:
        print(f"{perspective}: {len(perspective_data[perspective])}")
    
    # Sample data
    sampled_data = []
    sampling_stats = {}
    
    for perspective in perspectives:
        available_data = perspective_data[perspective]
        
        if not available_data:
            print(f"Warning: no data available for {perspective}")
            sampling_stats[perspective] = 0
            continue
        
        # Deduplicate by paper: at most one sample per paper
        paper_to_samples = defaultdict(list)
        for sample in available_data:
            paper_to_samples[sample["paper_id"]].append(sample)
        
        # Select the highest-confidence sample for each paper
        unique_samples = []
        for paper_id, samples in paper_to_samples.items():
            best_sample = max(samples, key=lambda x: x["confidence_score"])
            unique_samples.append(best_sample)
        
        # Random sampling
        sample_count = min(samples_per_perspective, len(unique_samples))
        selected_samples = random.sample(unique_samples, sample_count)
        
        # Convert to ShareGPT format
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
            print(f"Warning: only collected {sample_count} samples for {perspective}, fewer than the target {samples_per_perspective}")
    
    print(f"\nSampling summary:")
    total_samples = 0
    for perspective in perspectives:
        count = sampling_stats.get(perspective, 0)
        print(f"{perspective}: {count}")
        total_samples += count
    
    print(f"Total: {total_samples}")
    
    return sampled_data

def main():
    """Main function"""
    print(f"Starting SFT dataset generation...")
    print(f"Input file: {JSONL_FILE}")
    print(f"Paper directory: {PAPER_MD_DIR}")
    print(f"Samples per perspective: {SAMPLES_PER_PERSPECTIVE}")
    print(f"Minimum confidence: {MIN_CONFIDENCE}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Check input paths
    if not os.path.exists(JSONL_FILE):
        print(f"Error: input file not found: {JSONL_FILE}")
        return
    
    if not os.path.exists(PAPER_MD_DIR):
        print(f"Error: paper directory not found: {PAPER_MD_DIR}")
        return
    
    # Load data
    print("Loading data...")
    data = load_jsonl(JSONL_FILE)
    print(f"Loaded data for {len(data)} papers")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Sample data
    sampled_data = sample_data_by_perspective(
        data, 
        PAPER_MD_DIR, 
        samples_per_perspective=SAMPLES_PER_PERSPECTIVE,
        min_confidence=MIN_CONFIDENCE
    )
    
    # Save results
    print(f"\nSaving data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset generation complete! Total samples: {len(sampled_data)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()