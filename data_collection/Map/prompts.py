"""
Prompt Templates for Review-Rebuttal Mapping Pipeline

This module contains all the prompt templates used for segmenting review weaknesses
and mapping them to rebuttal responses.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class WeaknessPoint:
    """Represents a single weakness point"""
    id: str
    content: str


class PromptBuilder:
    """Class for building prompts for the review-rebuttal mapping pipeline"""
    
    @staticmethod
    def build_segmentation_messages(weaknesses_text: str) -> List[Dict[str, str]]:
        """Build messages for segmenting weaknesses into individual points"""
        
        system_message = """You are a professional academic review text analysis assistant. Your task is to segment a complete "Weaknesses & Questions" section from an academic paper review into independent, specific points.

You must follow these rules:
1. Each point should be an independent, specific issue or weakness
2. Preserve the core meaning of the original text without adding or removing information
3. Maintain existing numbering structures if present (e.g., 1., 2., W1, W2, etc.)
4. Handle various formatting styles including:
   - Numbered lists (1., 2., 3.)
   - Letter prefixes (W1, W2, Q1, Q2)
   - Markdown bullet points (-, *, +)
   - Section headers (## Weaknesses, ## Questions)
5. If no clear numbering exists, logically segment based on content structure
6. Each point should contain sufficient context to be understood independently
7. Preserve the original language and terminology used by the reviewer"""

        user_message = f"""Please segment the following Weaknesses & Questions text into independent points:

{weaknesses_text}

IMPORTANT: Regardless of the input format (bullet points, numbered lists, paragraphs, etc.), you MUST output in this exact format:

Point 1: [Complete content of the first weakness point]
Point 2: [Complete content of the second weakness point]
Point 3: [Complete content of the third weakness point]
...

Rules:
- Use exactly "Point X:" where X is a number starting from 1
- Include ALL weakness points from the input, don't skip any
- Each point should be complete and independently understandable
- Preserve the original meaning and wording as much as possible
- If the input has bullet points (-, *, +) or numbered lists (1., 2.), convert them to Point format
- If the input has long paragraphs, break them into logical points"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    @staticmethod
    def build_mapping_messages(weakness_points: List[WeaknessPoint], rebuttal_text: str) -> List[Dict[str, str]]:
        """Build messages for mapping weakness points to rebuttal responses"""
        
        system_message = """You are a professional academic review analysis assistant. Your task is to perform precise one-to-one mapping between review weakness points and author rebuttal responses.

Guidelines for mapping:
1. Carefully analyze the rebuttal text to identify which sections respond to specific weaknesses
2. Look for explicit references (W1, W2, Point 1, etc.) or implicit topical connections
3. Extract the complete response content that addresses each weakness
4. Assign confidence scores (0-1) based on the clarity and directness of the mapping
5. Mark as "No Response" if a weakness is not addressed in the rebuttal
6. Be conservative with confidence scores - only use high scores (>0.8) when the mapping is very clear
7. Preserve the exact wording from the rebuttal when extracting responses

CRITICAL RULE - NO SHORTCUTS OR REFERENCES:
You must NEVER use summarizing phrases or references like "[Same content as W2 response]", "[Similar to above]", "[As mentioned earlier]", etc. Always copy the complete, verbatim text from the rebuttal for each weakness point, even if the same rebuttal section addresses multiple weaknesses. Repetition is required and expected - do not try to avoid it."""

        # Format weakness points
        weakness_list = "\n".join([f"W{i+1}: {point.content}" for i, point in enumerate(weakness_points)])
        
        user_message = f"""Please map each weakness point in `<weakness_points>` to its corresponding rebuttal response from `<rebuttal_text>`:

<weakness_points>
{weakness_list}
</weakness_points>

<rebuttal_text>
{rebuttal_text}
</rebuttal_text>

MANDATORY REQUIREMENTS:
- Output a mapping line for EVERY weakness point (W1, W2, W3, ... up to W{len(weakness_points)})
- Use exactly "W[number] -> R[number]:" or "W[number] -> No Response" format
- Include confidence score in parentheses for every mapping
- Do not skip any weakness numbers
- If you cannot find a rebuttal response, write "No Response" instead of omitting the line

CRITICAL: You MUST provide a mapping for EVERY weakness point listed above. Do not skip any weakness points.
ABSOLUTELY CRITICAL - NO SUMMARIZATION OR SHORTCUTS FOR REBUTTAL CONTENT:
- ALWAYS copy the COMPLETE, FULL, VERBATIM text from the rebuttal for each weakness, even if content is identical to previous responses
- NEVER use summarizing or abbreviated phrases like "[Same content as W2 response]" or "[Similar to above]" or "[Full content identical to W5 response]" - always provide the complete original text
- Do NOT abbreviate, summarize, or reference other responses
- If the same rebuttal section addresses multiple weaknesses, copy the ENTIRE text in full for each relevant weakness
- NEVER add any commentary, explanation, or meta-text beyond the actual rebuttal content

</output_format> (use EXACTLY this format):
W1 -> R1: [Specific rebuttal content addressing W1] (Confidence: 0.xx)
W2 -> R2: [Specific rebuttal content addressing W2] (Confidence: 0.xx)
W3 -> R3: [Specific rebuttal content addressing W3] (Confidence: 0.xx)
...continue for ALL weakness points...

If no rebuttal response exists for a weakness:
Wx -> No Response (Confidence: 1.0)
</output_format>"""


        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
