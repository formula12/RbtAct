#!/usr/bin/env python3
"""
Review-Rebuttal Point-to-Point Mapping Pipeline

This script implements a complete pipeline for point-to-point mapping between 
weaknesses & questions in reviews and rebuttal text.

Pipeline contains two main steps:
1. Segment: Split complete weaknesses_and_questions into independent points
2. Map: Map each weakness point to corresponding responses in rebuttal_text

Author: Assistant
Date: 2025-09-16
"""

import json
import asyncio
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from openai_utils import process_data, client_dict
from prompts import PromptBuilder, WeaknessPoint


@dataclass
class RebuttalResponse:
    """Represents a rebuttal response"""
    id: str
    content: str
    

@dataclass
class WeaknessRebuttalMapping:
    """Represents mapping between weakness point and rebuttal response"""
    weakness_point: WeaknessPoint
    rebuttal_response: RebuttalResponse
    confidence_score: float


class ReviewRebuttalMapper:
    """Main class for Review-Rebuttal mapping pipeline"""
    
    def __init__(self, model_name: str = "gpt-5-chat"):
        self.model_name = model_name
        self.max_tokens = 8192
        self.prompt_builder = PromptBuilder()
    
    def parse_segmentation_result(self, result_text: str) -> List[WeaknessPoint]:
        """Parse segmentation result text to extract weakness points"""
        points = []
        
        # print(f"  Parsing segmentation result (length: {len(result_text)} chars)")
        
        # 首先尝试标准的 "Point X:" 格式
        pattern = r'Point\s+(\d+):\s*(.*?)(?=Point\s+\d+:|$)'
        matches = re.findall(pattern, result_text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # print(f"  Found {len(matches)} points using Point X: regex")
            for point_id, content in matches:
                clean_content = content.strip()
                if clean_content:
                    points.append(WeaknessPoint(
                        id=f"P{point_id}",
                        content=clean_content
                    ))
                    # print(f"    Point {point_id}: {clean_content[:50]}...")
        else:
            # try other formats
            # print("  No Point X: format found, trying alternative formats")
            
            # try numbered list format "1. 2. 3."
            pattern_numbered = r'(\d+)\.\s*(.*?)(?=\d+\.|$)'
            numbered_matches = re.findall(pattern_numbered, result_text, re.DOTALL)
            
            if numbered_matches:
                # print(f"  Found {len(numbered_matches)} points using numbered list regex")
                for point_id, content in numbered_matches:
                    clean_content = content.strip()
                    if clean_content:
                        points.append(WeaknessPoint(
                            id=f"P{point_id}",
                            content=clean_content
                        ))
                        # print(f"    Point {point_id}: {clean_content[:50]}...")
            else:
                # try bullet points format
                pattern_bullet = r'^[-*+]\s*(.*?)(?=^[-*+]|$)'
                bullet_matches = re.findall(pattern_bullet, result_text, re.MULTILINE | re.DOTALL)
                
                if bullet_matches:
                    # print(f"  Found {len(bullet_matches)} points using bullet point regex")
                    for i, content in enumerate(bullet_matches, 1):
                        clean_content = content.strip()
                        if clean_content:
                            points.append(WeaknessPoint(
                                id=f"P{i}",
                                content=clean_content
                            ))
                            # print(f"    Point {i}: {clean_content[:50]}...")
                else:
                    # try last resort: split by paragraphs
                    # print("  No structured format found, trying paragraph splitting")
                    paragraphs = [p.strip() for p in result_text.split('\n\n') if p.strip()]
                    for i, paragraph in enumerate(paragraphs, 1):
                        if len(paragraph) > 20:  # filter out too short paragraphs
                            points.append(WeaknessPoint(
                                id=f"P{i}",
                                content=paragraph
                            ))
                            # print(f"    Paragraph {i}: {paragraph[:50]}...")
        
        # print(f"  Total points extracted: {len(points)}")
        return points
    
    def parse_mapping_result(self, result_text: str, weakness_points: List[WeaknessPoint]) -> List[WeaknessRebuttalMapping]:
        """Parse mapping result text to extract mapping relationships"""
        mappings = []
        
        # print(f"  Parsing mapping result (length: {len(result_text)} chars)")
        # print(f"  Available weakness points: {[p.id for p in weakness_points]}")
        
        # First, let's see what the LLM actually output
        lines = result_text.split('\n')
        # print(f"  LLM output sample (first 10 lines):")
        # for i, line in enumerate(lines[:10]):
        #     if line.strip():
        #         print(f"    Line {i+1}: {line.strip()[:100]}...")
        
        # Try multiple regex patterns to catch different formats
        patterns = [
            # "No Response" format: W1 -> No Response (Confidence: 0.xx)
            r'W(\d+)\s*->\s*(No Response)\s*\(Confidence:\s*([\d.]+)\)',
            # Standard format: W1 -> R1: content (Confidence: 0.xx)
            r'W(\d+)\s*->\s*R\d+:\s*(.*?)\s*\(Confidence:\s*([\d.]+)\)',
            # Alternative format: W1 -> content (Confidence: 0.xx)
            r'W(\d+)\s*->\s*(.*?)\s*\(Confidence:\s*([\d.]+)\)',
            # Format without R: W1: content (Confidence: 0.xx)
            r'W(\d+):\s*(.*?)\s*\(Confidence:\s*([\d.]+)\)',
            # Format with different arrow: W1 => content (Confidence: 0.xx)
            r'W(\d+)\s*=>\s*(.*?)\s*\(Confidence:\s*([\d.]+)\)',
        ]
        
        matched_weaknesses = set()
        
        for pattern_idx, pattern in enumerate(patterns):
            matches = re.findall(pattern, result_text, re.DOTALL | re.IGNORECASE)
            
            if matches:
                # print(f"  Pattern {pattern_idx + 1} found {len(matches)} matches")
                
                for weakness_id, rebuttal_content, confidence in matches:
                    # Skip if we already matched this weakness
                    if weakness_id in matched_weaknesses:
                        continue
                        
                    try:
                        # Find the weakness point with matching ID
                        weakness_point = None
                        for point in weakness_points:
                            if point.id == f"P{weakness_id}":
                                weakness_point = point
                                break
                        
                        if weakness_point:
                            # Clean up the rebuttal content
                            clean_content = rebuttal_content.strip()
                            
                            # Handle "No Response" case - check multiple conditions
                            if (clean_content == "No Response" or 
                                clean_content.lower() in ["no response", "none", "not addressed", ""] or
                                len(clean_content) == 0):
                                rebuttal_response = RebuttalResponse(
                                    id="NO_RESPONSE",
                                    content="No Response"
                                )
                            else:
                                # Remove any leading "R[number]:" if present
                                clean_content = re.sub(r'^R\d+:\s*', '', clean_content)
                                # Final check after cleaning
                                if len(clean_content.strip()) == 0:
                                    rebuttal_response = RebuttalResponse(
                                        id="NO_RESPONSE", 
                                        content="No Response"
                                    )
                                else:
                                    rebuttal_response = RebuttalResponse(
                                        id=f"R{weakness_id}",
                                        content=clean_content.strip()
                                    )
                            
                            mappings.append(WeaknessRebuttalMapping(
                                weakness_point=weakness_point,
                                rebuttal_response=rebuttal_response,
                                confidence_score=float(confidence)
                            ))
                            
                            matched_weaknesses.add(weakness_id)
                            # print(f"    Mapped W{weakness_id} -> R{weakness_id} (confidence: {confidence})")
                        else:
                            # print(f"    Warning: Could not find weakness point P{weakness_id}")
                            pass
                            
                    except (ValueError, IndexError) as e:
                        # print(f"    Error parsing mapping W{weakness_id}: {e}")
                        continue
            else:
                # print(f"  Pattern {pattern_idx + 1} found no matches")
                pass
        
        # Report missing mappings
        expected_weaknesses = {str(i+1) for i in range(len(weakness_points))}
        missing_weaknesses = expected_weaknesses - matched_weaknesses
        
        # Debug information is now stored in debug_info, not printed to console
        
        return mappings
    
    async def segment_weaknesses(self, weaknesses_text: str) -> Tuple[List[WeaknessPoint], Dict]:
        """Segment weaknesses text into individual points"""
        
        def build_messages(example: Dict) -> List[Dict[str, str]]:
            return self.prompt_builder.build_segmentation_messages(example["weaknesses_text"])
        
        def post_process(example: Dict, response: str, model: str) -> Dict:
            example["segmentation_result"] = response
            return example
        
        # Prepare data
        data = [{"weaknesses_text": weaknesses_text}]
        
        # Call LLM for segmentation
        results = await process_data(
            data=data,
            client_dict=client_dict,
            build_messages=build_messages,
            response_model=None,  # Use unstructured output
            models=[self.model_name],
            enable_structured_output=False,
            post_process=post_process,
            max_concurrency=1,
            max_tokens=self.max_tokens
        )
        
        if results and results[0] and "segmentation_result" in results[0]:
            segmentation_text = results[0]["segmentation_result"]
            debug_info = {
                "input_weaknesses_length": len(weaknesses_text),
                "llm_output_length": len(segmentation_text),
                "llm_raw_output": segmentation_text,
                "step": "segmentation"
            }
            parsed_points = self.parse_segmentation_result(segmentation_text)
            debug_info["parsed_points_count"] = len(parsed_points)
            return parsed_points, debug_info
        else:
            debug_info = {
                "error": "Segmentation step failed - no results returned",
                "step": "segmentation"
            }
            return [], debug_info
    
    async def map_weakness_to_rebuttal(self, weakness_points: List[WeaknessPoint], rebuttal_text: str) -> Tuple[List[WeaknessRebuttalMapping], Dict]:
        """Map weakness points to rebuttal responses"""
        
        def build_messages(example: Dict) -> List[Dict[str, str]]:
            return self.prompt_builder.build_mapping_messages(example["weakness_points"], example["rebuttal_text"])
        
        def post_process(example: Dict, response: str, model: str) -> Dict:
            example["mapping_result"] = response
            return example
        
        # Prepare data
        data = [{"weakness_points": weakness_points, "rebuttal_text": rebuttal_text}]
        
        # Call LLM for mapping
        results = await process_data(
            data=data,
            client_dict=client_dict,
            build_messages=build_messages,
            response_model=None,  # Use unstructured output
            models=[self.model_name],
            enable_structured_output=False,
            post_process=post_process,
            max_concurrency=1,
            max_tokens=self.max_tokens
        )
        
        if results and results[0] and "mapping_result" in results[0]:
            mapping_text = results[0]["mapping_result"]
            debug_info = {
                "number_of_weakness_points": len(weakness_points),
                "expected_weakness_ids": [p.id for p in weakness_points],
                "rebuttal_text_length": len(rebuttal_text),
                "llm_output_length": len(mapping_text),
                "llm_raw_output": mapping_text,
                "step": "mapping"
            }
            parsed_mappings = self.parse_mapping_result(mapping_text, weakness_points)
            debug_info["successfully_parsed_mappings"] = len(parsed_mappings)
            debug_info["expected_mappings"] = len(weakness_points)
            
            # Add detailed mapping analysis
            expected_weaknesses = {str(i+1) for i in range(len(weakness_points))}
            matched_weaknesses = {mapping.weakness_point.id.replace('P', '') for mapping in parsed_mappings}
            missing_weaknesses = expected_weaknesses - matched_weaknesses
            
            debug_info["mapping_analysis"] = {
                "expected_weaknesses": sorted(expected_weaknesses),
                "matched_weaknesses": sorted(matched_weaknesses),
                "missing_weaknesses": sorted(missing_weaknesses) if missing_weaknesses else [],
                "all_matched": len(missing_weaknesses) == 0
            }
            
            return parsed_mappings, debug_info
        else:
            debug_info = {
                "error": "Mapping step failed - no results returned",
                "step": "mapping"
            }
            return [], debug_info
    
    async def process_single_review(self, review_data: Dict) -> Optional[Dict]:
        """Process a single review through the complete pipeline"""
        
        # Extract required fields
        weaknesses_text = review_data.get("weaknesses_and_questions", "")
        rebuttal_text = review_data.get("rebuttal_text", "")
        
        if not weaknesses_text or not rebuttal_text:
            return None
        
        # Step 1: Segment weaknesses
        weakness_points, segmentation_debug = await self.segment_weaknesses(weaknesses_text)
        if not weakness_points:
            return None
        
        # Step 2: Map to rebuttal
        mappings, mapping_debug = await self.map_weakness_to_rebuttal(weakness_points, rebuttal_text)
        if not mappings:
            return None
        
        # Build output result
        result = {
            # Preserve original basic information
            "paper_id": review_data.get("paper_id"),
            "review_id": review_data.get("review_id"),
            "conference": review_data.get("conference"),
            "paper_title": review_data.get("paper_title"),
            
            # Add mapping results
            "weakness_rebuttal_mappings": [],
            
            # Add debug information
            "debug_info": {
                "segmentation": segmentation_debug,
                "mapping": mapping_debug
            }
        }
        
        # Sort mappings by weakness_point id to ensure P1, P2, P3... order
        sorted_mappings = sorted(mappings, key=lambda m: int(m.weakness_point.id[1:]))
        
        for mapping in sorted_mappings:
            result["weakness_rebuttal_mappings"].append({
                "weakness_point": {
                    "id": mapping.weakness_point.id,
                    "content": mapping.weakness_point.content
                },
                "rebuttal_response": {
                    "id": mapping.rebuttal_response.id,
                    "content": mapping.rebuttal_response.content
                },
                "confidence_score": mapping.confidence_score
            })
        
        return result
    
    async def process_dataset(self, input_file: str, output_file: str = None, max_samples: int = None) -> List[Dict]:
        """Process entire dataset"""
        
        # Read data
        print(f"Reading dataset: {input_file}")
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    # print(f"Error parsing line {i+1}: {e}")
                    continue
        
        print(f"Successfully read {len(data)} records")
        
        # Process each review
        results = []
        for i, review_data in enumerate(data):
            print(f"\nProgress: {i+1}/{len(data)}")
            result = await self.process_single_review(review_data)
            if result:
                results.append(result)
            
            # Add small delay to avoid API limits
            await asyncio.sleep(0.1)
        
        print(f"\nProcessing complete! Successfully processed {len(results)} reviews")
        
        # Save results
        if output_file:
            print(f"Saving results to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        return results


async def main():
    """Main function"""
    
    # Record start time
    start_time = time.time()
    
    print("=== Review-Rebuttal Mapping Pipeline ===")
    
    # Configure file paths
    input_file = "iclr2024_reviews_with_decision_rebuttal_structured_100_20250908_164304.jsonl"
    # input_file = "iclr2024_reviews_filtered_format_all_20250917_171508.jsonl"
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"weakness_rebuttal_mappings_{timestamp}.jsonl"
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Create mapper instance
    print("Creating mapper instance...")
    mapper = ReviewRebuttalMapper(model_name="gpt-5-chat")
    
    # Process dataset (first 2 samples for testing)
    print("Starting dataset processing...")
    try:
        results = await mapper.process_dataset(
            input_file=input_file,
            output_file=output_file,
            max_samples=2  # Process 2 samples for testing
        )
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate and output total runtime
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Program completed ===")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Total runtime: {total_time/60:.2f} minutes")
    if total_time > 3600:
        print(f"Total runtime: {total_time/3600:.2f} hours")


if __name__ == "__main__":
    asyncio.run(main())
