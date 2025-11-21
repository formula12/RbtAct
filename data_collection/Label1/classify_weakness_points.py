import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List
import sys
import os
from tqdm import tqdm

# Add the current directory to the path to import openai_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai_utils import process_single_example_unstructured, client_dict


# def build_classification_messages(example: Dict) -> List[Dict[str, str]]:
#     """Build messages for classification"""
#     weakness_content = example["weakness_point"]["content"]
    
#     prompt = f"""You are an expert in academic paper review classification. Your task is to classify the weakness point into one of the following 5 categories:

# 1. **Experiments**: Missing or insufficient experiments, lack of ablation studies, weak baseline comparisons, unclear dataset description or splits, inadequate hyperparameter sensitivity, inappropriate evaluation metrics

# 2. **Writing & Presentation**: Grammar, clarity, readability, ambiguous phrasing, typos, missing definitions of symbols/terms, unclear explanations of concepts; Figures, tables, and organization issues — unclear plots, missing legends, poor formatting, misplaced content, overall paper structure making it hard to follow

# 3. **Theory**: Incorrect mathematical derivations, flawed assumptions, weak theoretical justification, missing proofs, inconsistency between claims and formulas

# 4. **Novelty**: Lack of novelty or originality, overlap with prior work, incremental contribution, insufficient differentiation from existing methods

# 5. **Reproducibility**: Missing implementation details, absent code or pseudo-code, hyperparameters not specified, insufficient information to reproduce results

# Please analyze the following weakness point and classify it into one of the above categories. Respond with ONLY the category name (Experiments, Writing & Presentation, Theory, Novelty, or Reproducibility).

# Weakness point to classify:
# {weakness_content}

# Classification:"""

#     return [{"role": "user", "content": prompt}]

def build_classification_messages(example: Dict) -> List[Dict[str, str]]:
    """Build messages for classification"""
    weakness_content = example["weakness_point"]["content"]
    
    prompt = f"""You are an expert in academic paper review classification. Your task is to identify from which perspective the reviewer is raising concerns or questions about the paper.

The following review point is a weakness or question raised by a reviewer during peer review. Please classify this review point based on the PERSPECTIVE from which the reviewer is critiquing or questioning the paper:

1. **Experiments**: The reviewer is questioning experimental **setup and design**. This includes missing or insufficient experiments, lack of ablation studies, weak baseline comparisons, unclear descriptions of datasets, or issues with hyperparameter selection.

2. **Writing**: The reviewer is concerned about writing quality - grammar, clarity, readability, ambiguous phrasing, typos, missing definitions of symbols/terms, unclear explanations of concepts.

3. **Presentation**: The reviewer is critiquing presentation and organization - figures, tables, and organization issues, unclear plots, missing legends, poor formatting, misplaced content, overall paper structure making it hard to follow.

4. **Theory**: The reviewer is questioning theoretical aspects - incorrect mathematical derivations, flawed assumptions, weak theoretical justification, missing proofs, inconsistency between claims and formulas.

5. **Novelty**: The reviewer is questioning novelty and originality - lack of novelty or originality, overlap with prior work, incremental contribution, insufficient differentiation from existing methods.

6. **Reproducibility**: The reviewer is concerned about reproducibility - missing implementation details, absent code or pseudo-code, hyperparameters not specified, insufficient information to reproduce results.

7. **Evaluation**: The reviewer is concerned about how the experimental results are **measured, analyzed, and interpreted**. This includes the use of inappropriate or missing evaluation metrics, insufficient analysis of results, or inconsistencies between reported results and the paper's claims.

8.  **Miscellaneous**: Content that is not a direct review point (weaknesses, questions, suggestions) about the paper. This includes polite remarks, Summative or transitional comments, summaries of the paper's or review's content, or irrelevant text.

Please analyze the following review point and identify from which perspective the reviewer is raising their concern. Respond with ONLY the category name (Experiments, Writing, Presentation, Theory, Novelty, Reproducibility, Evaluation, Miscellaneous).

Review point to classify:
{weakness_content}

Perspective:"""

    return [{"role": "user", "content": prompt}]


def extract_perspective_from_response(response: str, default_count: Dict[str, int]) -> str:
    """Extract classification result from LLM response"""
    # Clean response text
    response = response.strip()
    
    # Define valid classification labels
    # valid_categories = [
    #     "Experiments",
    #     "Writing & Presentation", 
    #     "Theory",
    #     "Novelty",
    #     "Reproducibility"
    # ]
    valid_categories = [
        "Experiments",
        "Writing", 
        "Presentation", 
        "Theory",
        "Novelty",
        "Reproducibility",
        "Evaluation",
        "Miscellaneous"
    ]
    # Try direct matching
    for category in valid_categories:
        if category.lower() in response.lower():
            return category
    
    # If no match found, increment counter and return default
    default_count["count"] += 1
    default_count["responses"].append(response)
    return "Miscellaneous"  # Default classification


def post_process_classification(example: Dict, response: str, model: str, default_count: Dict[str, int]) -> Dict:
    """Post-processing function to extract classification result and add to weakness_point"""
    perspective = extract_perspective_from_response(response, default_count)
    
    # Add perspective field to weakness_point
    example["weakness_point"]["perspective"] = perspective
    
    return example


async def process_weakness_points(input_file: str, output_file: str):
    """Main function for processing weakness point classification"""
    print(f"📖 Reading from {input_file}...")
    
    # Read input data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"📄 Loaded {len(data)} papers")
    
    # Expand all weakness_points for processing
    examples_to_process = []
    for paper_data in data:
        for mapping in paper_data.get("weakness_rebuttal_mappings", []):
            if "weakness_point" in mapping:
                # Create example with complete context
                example = {
                    "paper_id": paper_data["paper_id"],
                    "review_id": paper_data["review_id"],
                    "conference": paper_data["conference"],
                    "paper_title": paper_data["paper_title"],
                    "weakness_point": mapping["weakness_point"],
                    "original_mapping": mapping  # Keep original mapping reference
                }
                examples_to_process.append(example)
    
    print(f"🎯 Found {len(examples_to_process)} review points to classify")
    print("🚀 Starting classification process...\n")
    
    # Process all examples with progress bar
    processed_examples = []
    failed_count = 0
    default_count = {"count": 0, "responses": []}  # Counter: track default classifications and responses
    
    with tqdm(total=len(examples_to_process), desc="Classifying review points", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for i, example in enumerate(examples_to_process):
            try:
                result = await process_single_example_unstructured(
                    example,
                    client_dict=client_dict,
                    build_messages=build_classification_messages,
                    # model="gpt-5-chat",
                    model="gpt-5-mini",
                    post_process=lambda ex, resp, model: post_process_classification(ex, resp, model, default_count),
                    temperature=1.0,
                    max_tokens=2048,
                    # max_concurrency=1
                )
                
                if result is not None:
                    processed_examples.append(result)
                else:
                    failed_count += 1
                    # If processing fails, add default classification
                    example["weakness_point"]["perspective"] = "Miscellaneous"
                    processed_examples.append(example)
                    
            except Exception as e:
                failed_count += 1
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"\n❌ Error processing example ({error_type}): {error_msg}")
                
                # Check if it's a quota-related error
                if "quota" in error_msg.lower() or "rate limit" in error_msg.lower() or "429" in error_msg:
                    print(f"🚨 API quota may be exhausted! Please check Azure quota status")
                elif "401" in error_msg or "unauthorized" in error_msg.lower():
                    print(f"🔑 API key authentication failed! Please check Azure API key")
                elif "403" in error_msg or "forbidden" in error_msg.lower():
                    print(f"🚫 API access denied! Please check Azure service permissions")
                
                # If error occurs, add default classification
                example["weakness_point"]["perspective"] = "Miscellaneous"
                processed_examples.append(example)
            
            pbar.update(1)
    
    if failed_count > 0:
        print(f"\n⚠️  Warning: {failed_count} review points failed to process and were assigned default classification")
    
    if default_count["count"] > 0:
        print(f"\n🔍 Found {default_count['count']} responses that couldn't be matched to any category and were assigned default classification:")
        for i, response in enumerate(default_count["responses"], 1):
            print(f"  {i}. {response[:100]}{'...' if len(response) > 100 else ''}")
    
    # Reorganize data structure
    print("\n📊 Reorganizing results...")
    paper_results = {}
    
    for processed in processed_examples:
        paper_id = processed["paper_id"]
        review_id = processed["review_id"]
        key = (paper_id, review_id)
        
        if key not in paper_results:
            paper_results[key] = {
                "paper_id": paper_id,
                "review_id": review_id,
                "conference": processed["conference"],
                "paper_title": processed["paper_title"],
                "weakness_rebuttal_mappings": []
            }
        
        # Update weakness_point in original mapping
        original_mapping = processed["original_mapping"]
        original_mapping["weakness_point"] = processed["weakness_point"]
        
        paper_results[key]["weakness_rebuttal_mappings"].append(original_mapping)
    
    # Write to output file
    print(f"💾 Writing results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper_data in paper_results.values():
            f.write(json.dumps(paper_data, ensure_ascii=False) + '\n')
    
    print(f"✅ Classification completed! Results saved to {output_file}")
    
    # Print classification statistics
    category_counts = {}
    for processed in processed_examples:
        perspective = processed["weakness_point"].get("perspective", "Unknown")
        category_counts[perspective] = category_counts.get(perspective, 0) + 1
    
    print("\n📈 Classification Statistics:")
    for category, count in sorted(category_counts.items()):
        print(f"  📋 {category}: {count}")


async def main():
    """Main function"""
    # Input and output file paths
    # input_file = "weakness_rebuttal_mappings_20250918_123817.jsonl"
    input_file = "weakness_rebuttal_mappings_20250918_124157_20.jsonl"
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"iclr2024_map_per_{timestamp}.jsonl"
    
    print("🎯 Starting review point perspective classification...")
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    print("=" * 60)
    
    try:
        await process_weakness_points(input_file, output_file)
    except Exception as e:
        print(f"❌ Error in main process: {e}")
        raise


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())
