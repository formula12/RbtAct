#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset generation configuration file
"""

# File path configuration
JSONL_FILE = "iclr2024_map_merged_per_filter2_test.jsonl"
# JSONL_FILE = "iclr2024_map_merged_per_filter2_train.jsonl"
PAPER_MD_DIR = "paper_md/iclr2024"
OUTPUT_FILE = "test_dataset_60_noli.json"
# OUTPUT_FILE = "sft_dataset_sharegpt_test.json"

# Sampling configuration
SAMPLES_PER_PERSPECTIVE = 60  # Number of samples per perspective
MIN_CONFIDENCE = 0.9  # Minimum confidence threshold
RANDOM_SEED = 42  # Random seed

# Supported perspective list
PERSPECTIVES = [
    "Experiments", 
    "Theory", 
    "Evaluation", 
    "Novelty", 
    "Writing", 
    "Presentation", 
    "Reproducibility"
]

# Perspective mapping (for generating user request text)
PERSPECTIVE_MAP = {
    "Experiments": "experiments",
    "Theory": "theory",
    "Evaluation": "evaluation", 
    "Novelty": "novelty",
    "Writing": "writing",
    "Presentation": "presentation",
    "Reproducibility": "reproducibility"
}

# System prompt
SYSTEM_PROMPT = "You are a professional reviewer. Your task is to provide a constructive comment on the given scientific paper from a specific perspective."

# User request template
USER_REQUEST_TEMPLATE = """[REQUEST]: From the perspective of {perspective_text}, provide a constructive comment on the following paper.

[BEGIN PAPER]
{paper_content}
[END PAPER]"""

