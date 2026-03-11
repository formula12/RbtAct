#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the generated dataset format
"""

import json
from pathlib import Path

def test_dataset_format(dataset_file: str):
    """Test whether the dataset format is correct"""
    
    if not Path(dataset_file).exists():
        print(f"Dataset file {dataset_file} does not exist")
        return False
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return False
    
    if not isinstance(data, list):
        print("Dataset should be a list")
        return False
    
    print(f"Dataset contains {len(data)} samples")
    
    # Check the format of the first few samples
    for i, sample in enumerate(data[:3]):
        print(f"\n=== Sample {i+1} ===")
        
        if not isinstance(sample, dict):
            print(f"Sample {i+1} is not a dictionary")
            return False
        
        # Check required fields
        required_fields = ["conversations", "conference", "paper_id", "paper_title", "review_id", "perspective"]
        for field in required_fields:
            if field not in sample:
                print(f"Sample {i+1} is missing field: {field}")
                return False
        
        conversations = sample["conversations"]
        if not isinstance(conversations, list) or len(conversations) != 3:
            print(f"Sample {i+1} conversations should contain 3 messages")
            return False
        
        # Check roles
        expected_roles = ["system", "human", "gpt"]
        for j, msg in enumerate(conversations):
            if msg.get("from") != expected_roles[j]:
                print(f"Sample {i+1} message {j+1} has wrong role, expected {expected_roles[j]}")
                return False
            
            if not msg.get("value"):
                print(f"Sample {i+1} message {j+1} is empty")
                return False
        
        # Print sample metadata
        print(f"Conference: {sample['conference']}")
        print(f"Paper ID: {sample['paper_id']}")
        print(f"Paper Title: {sample['paper_title'][:50]}..." if len(sample['paper_title']) > 50 else f"Paper Title: {sample['paper_title']}")
        print(f"Review ID: {sample['review_id']}")
        print(f"Perspective: {sample['perspective']}")
        
        # Print sample content (truncated)
        print(f"System: {conversations[0]['value']}")
        user_content = conversations[1]['value']
        if len(user_content) > 200:
            user_content = user_content[:200] + "..."
        print(f"Human: {user_content}")
        
        assistant_content = conversations[2]['value']
        if len(assistant_content) > 200:
            assistant_content = assistant_content[:200] + "..."
        print(f"GPT: {assistant_content}")
    
    print(f"\n✅ Dataset format check passed!")
    return True

def analyze_perspectives(dataset_file: str):
    """Analyze the distribution of perspectives in the dataset"""
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    perspective_count = {}
    conference_count = {}
    
    for sample in data:
        # Read directly from the perspective field
        perspective = sample.get("perspective", "Unknown")
        perspective_count[perspective] = perspective_count.get(perspective, 0) + 1
        
        # Count conference distribution
        conference = sample.get("conference", "Unknown")
        conference_count[conference] = conference_count.get(conference, 0) + 1
    
    print(f"\n=== Perspective Distribution ===")
    for perspective, count in sorted(perspective_count.items()):
        print(f"{perspective}: {count}")
    
    print(f"\n=== Conference Distribution ===")
    for conference, count in sorted(conference_count.items()):
        print(f"{conference}: {count}")
    
    total = sum(perspective_count.values())
    print(f"\nTotal: {total}")

if __name__ == "__main__":
    dataset_file = "sft_dataset_sharegpt.json"
    
    if test_dataset_format(dataset_file):
        analyze_perspectives(dataset_file)
    else:
        print("Dataset format check failed")