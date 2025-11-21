# -*- coding: utf-8 -*-
import re, json, sys
from collections import OrderedDict
import openreview
from datetime import datetime

VENUE_ID = "ICLR.cc/2024/Conference"
N_SUBMISSIONS = 100     
BASEURL = "https://api2.openreview.net"

# Optional: Debug specific review
DEBUG_FORUM_ID  = "xxx"  
DEBUG_REVIEW_ID = "xxx"

# Rating field keywords (priority from high to low)
RATING_KEYS = [
    "overall_rating", "rating", "recommendation", "reviewer_rating",
    "overall_assessment", "overall", "score", "assessment"
]

# Structured field alias mapping
SECTION_ALIASES = {
    'summary': ['summary', 'abstract', 'overview'],
    'soundness': ['soundness'],
    'presentation': ['presentation', 'clarity', 'writing_quality'],
    'contribution': ['contribution', 'originality', 'novelty'],
    'strengths': ['strengths', 'pros', 'positive_aspects'],
    'weaknesses': ['weaknesses', 'cons', 'negative_aspects', 'concerns'],
    'questions': ['questions', 'questions_for_the_authors', 'additional_feedback', 'typos']
}

def _val(x):
    """Compatible with v2 field {'value': ...} or direct value"""
    if isinstance(x, dict):
        return x.get("value")
    return x

def parse_numeric_rating(v):
    """Parse '8: Accept' or '6' -> (8.0, 'Accept'); if parsing fails, return (None, text label or original)"""
    v = _val(v)
    if v is None:
        return None, None
    if isinstance(v, (int, float)):
        return float(v), None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None, None
        m = re.match(r'\s*([0-9]+(?:\.[0-9]+)?)\s*[:：]?\s*(.*)\s*$', s)
        if m:
            num = float(m.group(1))
            label = m.group(2).strip() or None
            return num, label
        # No number, pure label
        return None, s
    return None, None

def find_rating_field_name(content: dict):
    """Find the most likely rating field name in content keys"""
    keys = list(content.keys())
    lowmap = {k: k.lower() for k in keys}
    # Prioritize the list
    for pref in RATING_KEYS:
        for k in keys:
            if pref in lowmap[k]:
                return k
    # Fallback: any containing rating / score / recommend / assess
    for k in keys:
        lk = lowmap[k]
        if any(t in lk for t in ["rating", "score", "recommend", "assess"]):
            return k
    return None

def join_review_text(content: dict):
    """Merge all fields that appear to be 'main text' into one review_text"""
    texts = []
    for k, v in content.items():
        val = _val(v)
        if isinstance(val, str):
            s = val.strip()
            # Filter out obvious non-text fields: short labels/empty strings/ID-like fields
            if s and not re.fullmatch(r'[0-9.:/,_-]+', s) and len(s) >= 3:
                texts.append(f"## {k}\n{s}" if k.lower() not in ["review", "summary"] else s)
    # If there's a dedicated review field, use it exclusively
    if 'review' in content and isinstance(_val(content['review']), str):
        s = _val(content['review']).strip()
        return s if s else None
    return "\n\n".join(texts) if texts else None

def extract_rating_from_note_content(content: dict):
    """Extract rating from a note's content; returns (raw_value, num, label, field_name)"""
    if not isinstance(content, dict):
        return None, None, None, None
    fname = find_rating_field_name(content)
    if not fname:
        return None, None, None, None
    raw = _val(content.get(fname))
    num, label = parse_numeric_rating(raw)
    return raw, num, label, fname

def extract_confidence_from_content(content: dict):
    """Extract confidence score from content"""
    if not isinstance(content, dict):
        return None
    
    # Try different confidence field names
    confidence_fields = ['confidence', 'reviewer_confidence', 'confidence_rating']
    for field in confidence_fields:
        if field in content:
            val = _val(content[field])
            if val is not None:
                # Try parsing numeric score
                num, _ = parse_numeric_rating(val)
                return num
    return None

def extract_structured_fields_from_content(content: dict):
    """Extract structured fields from review content"""
    if not isinstance(content, dict):
        return {k: None for k in SECTION_ALIASES.keys()}
    
    result = {k: None for k in SECTION_ALIASES.keys()}
    
    # First try direct field matching
    for std_key, alias_list in SECTION_ALIASES.items():
        for alias in alias_list:
            if alias in content:
                val = _val(content[alias])
                if isinstance(val, str) and val.strip():
                    result[std_key] = val.strip()
                    break
    
    return result

def extract_structured_fields_from_text(review_text: str):
    """Parse structured fields from review text (fallback method)"""
    if not review_text:
        return {k: None for k in SECTION_ALIASES.keys()}
    
    result = {k: None for k in SECTION_ALIASES.keys()}
    lines = review_text.splitlines()
    current_section = None
    section_content = []
    
    for line in lines:
        line = line.strip()
        # Check if it's a section title
        section_match = None
        for std_key in SECTION_ALIASES.keys():
            if re.match(rf'^\s*##\s*{std_key}\s*$', line, re.IGNORECASE):
                section_match = std_key
                break
        
        if section_match:
            # Save previous section
            if current_section and section_content:
                result[current_section] = '\n'.join(section_content).strip() or None
            # Start new section
            current_section = section_match
            section_content = []
        elif current_section and line:
            section_content.append(line)
    
    # Save last section
    if current_section and section_content:
        result[current_section] = '\n'.join(section_content).strip() or None
    
    return result

def first_last_rating_via_edits(client, note_id: str):
    """Query edits: get first and last ratings (if exist)"""
    edits = client.get_note_edits(note_id=note_id, sort='tcdate:asc')
    first = last = None
    first_name = last_name = None
    if edits:
        # First version
        note0 = edits[0].note
        if note0 and isinstance(note0.content, dict):
            raw, num, label, fname = extract_rating_from_note_content(note0.content)
            if raw is not None:
                first = (raw, num, label); first_name = fname
        # Last version
        noteZ = edits[-1].note
        if noteZ and isinstance(noteZ.content, dict):
            raw, num, label, fname = extract_rating_from_note_content(noteZ.content)
            if raw is not None:
                last = (raw, num, label); last_name = fname
    return first, last, first_name, last_name, len(edits)

def earliest_and_latest_rating_via_edits(client, note_id: str):
    """Returns (first, last, first_name, last_name, n_edits)
    where first/last are tuples (raw, num, label); *first occurrence of rating* is initial score,
    *last occurrence of rating* is final score.
    """
    edits = client.get_note_edits(note_id=note_id, sort='tcdate:asc')
    first = last = None
    first_name = last_name = None

    for e in edits:
        note = e.note
        if not note or not isinstance(note.content, dict):
            continue
        raw, num, label, fname = extract_rating_from_note_content(note.content)
        if raw is None:
            # Field name may be different in this version, extract_rating_from_note_content has done fuzzy matching;
            # If still can't get it, skip
            continue
        if first is None:
            first = (raw, num, label)
            first_name = fname
        # Keep overwriting to ensure last is the "last occurrence of rating" version
        last = (raw, num, label)
        last_name = fname

    return first, last, first_name, last_name, len(edits)


def safe_note_title(n):
    t = n.content.get('title')
    return _val(t) if t else None

def is_official_review(reply):
    invs = reply.get('invitations', [])
    return any(x.endswith('Official_Review') for x in invs)

def is_decision(reply):
    """Check if it's a decision note"""
    invs = reply.get('invitations', [])
    return any(x.endswith('Decision') for x in invs)

def is_author_rebuttal(reply, venue_id):
    """Check if it's an author's rebuttal note (not a reviewer's reply)"""
    invs = reply.get('invitations', [])
    sigs = reply.get('signatures', [])
    
    # Method 1: Check if invitation name contains explicit author rebuttal identifier
    for inv in invs:
        inv_lower = inv.lower()
        # More precise pattern: must be official author response/rebuttal
        if any(pattern in inv_lower for pattern in [
            '/authors/-/official_comment',
            '/authors/-/rebuttal', 
            'author_response',
            'authors_official_comment'
        ]):
            return True
        # Check if invitation ends with Authors (official author response)
        if '/authors' in inv_lower and ('comment' in inv_lower or 'rebuttal' in inv_lower):
            return True
    
    # Method 2: Check signatures - author responses are usually signed by Authors group
    for sig in sigs:
        if sig.endswith('/Authors') and venue_id in sig:
            return True
    
    return False

def is_reviewer_discussion(reply, venue_id):
    """Check if it's a reviewer's discussion reply"""
    invs = reply.get('invitations', [])
    sigs = reply.get('signatures', [])
    
    # Exclude official review and author rebuttal
    if is_official_review(reply) or is_author_rebuttal(reply, venue_id):
        return False
    
    # Check invitation pattern - relax conditions
    for inv in invs:
        inv_lower = inv.lower()
        # Reviewer discussion usually has these patterns
        if any(pattern in inv_lower for pattern in [
            'official_comment',
            'comment',
            'discussion'
        ]) and not any(exclude in inv_lower for exclude in [
            'authors',
            'author'
        ]):
            return True
    
    # Check signatures - reviewers usually sign in a specific format
    for sig in sigs:
        # More lenient condition: contains venue_id and is not Authors signature
        if venue_id in sig and '/Authors' not in sig and '/Area_Chairs' not in sig:
            # Further check if it looks like a reviewer signature
            if 'Reviewer_' in sig or 'reviewer' in sig.lower():
                return True
    
    # If there's content indicating it's not an empty reply, and not author/AC, it may be reviewer discussion
    content = reply.get('content', {})
    if content and not any(pattern in str(sigs).lower() for pattern in ['authors', 'area_chairs']):
        # Check if there's text content
        for field in ['comment', 'text', 'message', 'response']:
            if field in content and content[field]:
                return True
    
    return False

def extract_reply_context(content):
    """Extract reply context information, such as 'Replying to XXX'"""
    if not isinstance(content, dict):
        return None
    
    # Try different field names to get reply content
    text_content = None
    for field in ['comment', 'response', 'text', 'message', 'rebuttal']:
        if field in content:
            text_content = _val(content[field])
            if text_content and isinstance(text_content, str):
                text_content = text_content.strip()
                break
    
    if not text_content:
        return None
    
    # Find reply context
    reply_context = {}
    
    # Find patterns like "Replying to", "@Reviewer", "Dear Reviewer"
    import re
    patterns = [
        r'replying\s+to\s+([^:\n]+)',
        r'@([^:\s\n]+)',
        r'dear\s+([^:\n,]+)',
        r'response\s+to\s+([^:\n]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_content[:200], re.IGNORECASE)
        if matches:
            reply_context['mentioned_reviewer'] = matches[0].strip()
            break
    
    return reply_context if reply_context else None

def extract_decision(replies):
    """Extract final_decision from replies"""
    for r in replies:
        if is_decision(r):
            content = r.get('content', {}) or {}
            decision = _val(content.get('decision'))
            return decision
    return None

def extract_all_discussions(replies, venue_id):
    """Extract all discussions (all replies from authors and reviewers), without distinguishing types"""
    all_discussions = []
    
    for r in replies:
        # Skip official reviews and decisions
        if is_official_review(r) or is_decision(r):
            continue
            
        content = r.get('content', {}) or {}
        signatures = r.get('signatures', [])
        
        # Try to get text content
        text_content = None
        for field in ['comment', 'response', 'text', 'message', 'rebuttal', 'discussion']:
            if field in content:
                text_content = _val(content[field])
                if text_content and isinstance(text_content, str):
                    text_content = text_content.strip()
                    break
        
        if text_content:
            # Determine if it's author or reviewer
            is_author = any('/Authors' in str(sig) for sig in signatures)
            
            # Extract reply context
            reply_context = extract_reply_context(content)
            
            discussion_entry = {
                'discussion_id': r.get('id'),
                'discussion_text': text_content,
                'created_at': r.get('tcdate') or r.get('cdate'),
                'replyto': r.get('replyto'),
                'signatures': signatures,
                'invitations': r.get('invitations', []),
                'author_type': 'author' if is_author else 'reviewer'  # New field: distinguish type
            }
            
            # Add reply context information
            if reply_context:
                discussion_entry.update(reply_context)
            
            all_discussions.append(discussion_entry)
    
    # Sort by creation time
    all_discussions.sort(key=lambda x: x['created_at'] or 0)
    return all_discussions

def extract_rebuttals(replies, venue_id):
    """Filter author rebuttals from all discussions (maintain backward compatibility)"""
    all_discussions = extract_all_discussions(replies, venue_id)
    rebuttals = []
    
    for d in all_discussions:
        if d['author_type'] == 'author':
            # Convert to rebuttals format
            rebuttal_entry = {
                'rebuttal_id': d['discussion_id'],
                'rebuttal_text': d['discussion_text'],
                'created_at': d['created_at'],
                'replyto': d['replyto'],
                'signatures': d['signatures']
            }
            
            # Add reply context information
            if 'mentioned_reviewer' in d:
                rebuttal_entry['mentioned_reviewer'] = d['mentioned_reviewer']
            
            rebuttals.append(rebuttal_entry)
    
    return rebuttals

def extract_reviewer_discussions(replies, venue_id):
    """Filter reviewer discussions from all discussions"""
    all_discussions = extract_all_discussions(replies, venue_id)
    discussions = []
    
    for d in all_discussions:
        if d['author_type'] == 'reviewer':
            # Convert to discussions format
            discussion_entry = {
                'discussion_id': d['discussion_id'],
                'discussion_text': d['discussion_text'],
                'created_at': d['created_at'],
                'replyto': d['replyto'],
                'signatures': d['signatures'],
                'invitations': d['invitations']
            }
            
            # Add reply context information
            if 'mentioned_reviewer' in d:
                discussion_entry['mentioned_reviewer'] = d['mentioned_reviewer']
            
            discussions.append(discussion_entry)
    
    return discussions

def find_rebuttals_for_review(rebuttals, review_id, forum_id, strict_mode=True):
    """Find corresponding rebuttals for a specific review
    
    Args:
        rebuttals: List of all rebuttals
        review_id: ID of the target review
        forum_id: Forum ID of the paper
        strict_mode: If True, only return rebuttals directly replying to this review;
                    If False, include general rebuttals when no direct replies are found
    """
    review_rebuttals = []
    
    for r in rebuttals:
        replyto = r['replyto']
        
        if replyto == review_id:
            # Directly replying to this review
            review_rebuttals.append(r)
    
    # If not strict mode and no direct replies found, consider general replies
    if not strict_mode and not review_rebuttals:
        for r in rebuttals:
            replyto = r['replyto']
            if replyto == forum_id:
                # General rebuttal to the entire paper
                review_rebuttals.append(r)
    
    return review_rebuttals

def find_all_discussions_for_review(all_discussions, review_id, forum_id):
    """Find all related discussions for a specific review (including nested replies)
    
    Args:
        all_discussions: List of all discussions (including author and reviewer)
        review_id: ID of the target review
        forum_id: Forum ID of the paper
    """
    related_discussions = []
    found_ids = set()  # Use set to avoid duplicates
    
    # Round 1: Find all discussions directly replying to the review
    for d in all_discussions:
        if d['replyto'] == review_id:
            related_discussions.append(d)
            found_ids.add(d['discussion_id'])
    
    # Multiple rounds of recursion: Find nested replies to already found discussions
    # Continue searching until no new replies are found
    changed = True
    while changed:
        changed = False
        for d in all_discussions:
            # If this discussion replied to one of the discussions we've already found
            if d['replyto'] in found_ids and d['discussion_id'] not in found_ids:
                related_discussions.append(d)
                found_ids.add(d['discussion_id'])
                changed = True
    
    # Sort by time
    related_discussions.sort(key=lambda x: x['created_at'] or 0)
    return related_discussions

def find_rebuttals_for_review(rebuttals, review_id, forum_id, strict_mode=True):
    """Find corresponding rebuttals for a specific review (maintain compatibility)"""
    # Simplified logic, since rebuttals are now filtered from all_discussions
    review_rebuttals = []
    
    for r in rebuttals:
        if r['replyto'] == review_id:
            review_rebuttals.append(r)
    
    return review_rebuttals

def find_discussions_for_review(discussions, review_id, forum_id, strict_mode=True):
    """Find corresponding reviewer discussions for a specific review (maintain compatibility)"""
    # Simplified logic, since discussions are now filtered from all_discussions
    review_discussions = []
    
    for d in discussions:
        if d['replyto'] == review_id:
            review_discussions.append(d)
    
    return review_discussions

def format_rebuttals_separated(rebuttals):
    """Format rebuttals into separated structure, keeping each rebuttal independent"""
    if not rebuttals:
        return []
    
    formatted_rebuttals = []
    for r in rebuttals:
        formatted_rebuttal = {
            'rebuttal_id': r['rebuttal_id'],
            'rebuttal_text': r['rebuttal_text'],
            'created_at': r['created_at'],
            'replyto': r['replyto'],
            'signatures': r.get('signatures', [])
        }
        
        # Add reply context information (if available)
        if 'mentioned_reviewer' in r:
            formatted_rebuttal['mentioned_reviewer'] = r['mentioned_reviewer']
        
        formatted_rebuttals.append(formatted_rebuttal)
    
    return formatted_rebuttals

def format_discussions_separated(discussions):
    """Format discussions into separated structure, keeping each discussion independent"""
    if not discussions:
        return []
    
    formatted_discussions = []
    for d in discussions:
        formatted_discussion = {
            'discussion_id': d['discussion_id'],
            'discussion_text': d['discussion_text'],
            'created_at': d['created_at'],
            'replyto': d['replyto'],
            'signatures': d.get('signatures', []),
            'invitations': d.get('invitations', [])
        }
        
        # Add reply context information (if available)
        if 'mentioned_reviewer' in d:
            formatted_discussion['mentioned_reviewer'] = d['mentioned_reviewer']
        
        formatted_discussions.append(formatted_discussion)
    
    return formatted_discussions

def join_rebuttals(rebuttals):
    """Merge multiple rebuttals into one text (retained for compatibility)"""
    if not rebuttals:
        return None
    
    if len(rebuttals) == 1:
        return rebuttals[0]['rebuttal_text']
    
    # Multiple rebuttals, merge in chronological order
    texts = []
    for i, r in enumerate(rebuttals, 1):
        text = r['rebuttal_text']
        if len(rebuttals) > 1:
            texts.append(f"## Rebuttal {i} (ID: {r['rebuttal_id']})\n{text}")
        else:
            texts.append(text)
    
    return "\n\n".join(texts)

def join_reviewer_discussions(discussions):
    """Merge multiple reviewer discussions into one text"""
    if not discussions:
        return None
    
    if len(discussions) == 1:
        return discussions[0]['discussion_text']
    
    # Multiple discussions, merge in chronological order
    texts = []
    for i, d in enumerate(discussions, 1):
        text = d['discussion_text']
        if len(discussions) > 1:
            texts.append(f"## Discussion {i} (ID: {d['discussion_id']})\n{text}")
        else:
            texts.append(text)
    
    return "\n\n".join(texts)

def main():
    client = openreview.api.OpenReviewClient(baseurl=BASEURL, username='xxx', password='xxx')
    # 1) Find submission invitation name
    venue_group = client.get_group(VENUE_ID)
    submission_name = venue_group.content['submission_name']['value']

    # 2) Fetch submissions (with replies)
    subs = client.get_all_notes(invitation=f"{VENUE_ID}/-/{submission_name}", details="replies")
    # subs = subs[:N_SUBMISSIONS]

    out = []
    debug_printed = False

    for sub in subs:
        paper_id = sub.forum or sub.id
        paper_title = safe_note_title(sub)
        replies = sub.details.get("replies", []) or []
        
        # Debug: print all invitation names and rebuttal information
        if paper_id == DEBUG_FORUM_ID:
            sys.stderr.write(f"\n=== DEBUG: All invitations for paper {paper_id} ===\n")
            for r in replies:
                invs = r.get('invitations', [])
                sigs = r.get('signatures', [])
                replyto = r.get('replyto')
                is_review = is_official_review(r)
                is_auth_rebuttal = is_author_rebuttal(r, VENUE_ID)
                is_reviewer_disc = is_reviewer_discussion(r, VENUE_ID)
                sys.stderr.write(f"Reply ID: {r.get('id')}, replyto: {replyto}\n")
                sys.stderr.write(f"  - Invitations: {invs}\n")
                sys.stderr.write(f"  - Signatures: {sigs}\n")
                sys.stderr.write(f"  - Is review: {is_review}, Is author rebuttal: {is_auth_rebuttal}, Is reviewer discussion: {is_reviewer_disc}\n")
            sys.stderr.flush()
        
        # Extract decision and all discussions once for each paper
        final_decision = extract_decision(replies)
        paper_all_discussions = extract_all_discussions(replies, VENUE_ID)  # New method: get all discussions
        paper_rebuttals = extract_rebuttals(replies, VENUE_ID)
        paper_discussions = extract_reviewer_discussions(replies, VENUE_ID)
        
        # Simple statistics
        decision_count = sum(1 for r in replies if is_decision(r))
        rebuttal_count = sum(1 for r in replies if is_author_rebuttal(r, VENUE_ID))
        discussion_count = sum(1 for r in replies if is_reviewer_discussion(r, VENUE_ID))
        
        for r in replies:
            if not is_official_review(r):
                continue

            rid = r["id"]
            rcontent = r.get("content", {}) or {}
            # Combine text
            review_text = join_review_text(rcontent)
            if review_text == "":
                review_text = None

            # First extract "initial/final" from edits
            first, last, first_name, last_name, n_edits = earliest_and_latest_rating_via_edits(client, rid)
        
            # If not found in edits, check current snapshot
            if first is None and last is None:
                raw_now, num_now, label_now, fname_now = extract_rating_from_note_content(rcontent)
                if raw_now is not None:
                    first = last = (raw_now, num_now, label_now)
                    first_name = last_name = fname_now

            def pack(name_triplet, which="initial"):
                if name_triplet is None:
                    return {
                        f"{which}_rating_raw": "no_rating",
                        f"{which}_rating": None,
                        f"{which}_label": None
                    }
                raw, num, label = name_triplet
                return {
                    f"{which}_rating_raw": raw if raw is not None else "no_rating",
                    f"{which}_rating": num,
                    f"{which}_label": label
                }

            # Extract structured fields
            structured_fields = extract_structured_fields_from_content(rcontent)
            # If direct field extraction fails, try parsing from text
            if all(v is None for v in structured_fields.values()) and review_text:
                structured_fields = extract_structured_fields_from_text(review_text)
            
            # Extract confidence
            confidence = extract_confidence_from_content(rcontent)

            row = OrderedDict()
            row['conference'] = 'ICLR-2024'
            row["paper_id"] = paper_id
            row["paper_title"] = paper_title
            row["review_id"] = rid
            row["review_text"] = review_text
            
            # Add structured fields
            # row["summary"] = structured_fields.get("summary")
            # row["soundness"] = structured_fields.get("soundness")
            # row["presentation"] = structured_fields.get("presentation")
            # row["contribution"] = structured_fields.get("contribution")
            # row["strengths"] = structured_fields.get("strengths")
            
            # Merge weaknesses and questions into one field
            weaknesses_text = structured_fields.get("weaknesses")
            questions_text = structured_fields.get("questions")
            
            combined_concerns = []
            if weaknesses_text and weaknesses_text.strip():
                combined_concerns.append(f"## Weaknesses\n{weaknesses_text.strip()}")
            if questions_text and questions_text.strip():
                combined_concerns.append(f"## Questions\n{questions_text.strip()}")
            
            row["weaknesses_and_questions"] = "\n\n".join(combined_concerns) if combined_concerns else None
            row["confidence"] = confidence
            
            # Only keep final_rating (numeric score)
            if last is not None:
                _, final_rating_num, _ = last
                row["final_rating"] = final_rating_num
            else:
                row["final_rating"] = None
            row["n_edits"] = n_edits
            
            # Find all related discussions for this specific review (including nested replies)
            all_related_discussions = find_all_discussions_for_review(paper_all_discussions, rid, paper_id)
            
            # Separate author and reviewer content from all related discussions
            review_author_discussions = [d for d in all_related_discussions if d['author_type'] == 'author']
            review_reviewer_discussions = [d for d in all_related_discussions if d['author_type'] == 'reviewer']
            
            # Convert to original format (maintain compatibility)
            review_specific_rebuttals = []
            for d in review_author_discussions:
                rebuttal_entry = {
                    'rebuttal_id': d['discussion_id'],
                    'rebuttal_text': d['discussion_text'],
                    'created_at': d['created_at'],
                    'replyto': d['replyto'],
                    'signatures': d['signatures']
                }
                if 'mentioned_reviewer' in d:
                    rebuttal_entry['mentioned_reviewer'] = d['mentioned_reviewer']
                review_specific_rebuttals.append(rebuttal_entry)
            
            review_specific_discussions = []
            for d in review_reviewer_discussions:
                discussion_entry = {
                    'discussion_id': d['discussion_id'],
                    'discussion_text': d['discussion_text'],
                    'created_at': d['created_at'],
                    'replyto': d['replyto'],
                    'signatures': d['signatures'],
                    'invitations': d['invitations']
                }
                if 'mentioned_reviewer' in d:
                    discussion_entry['mentioned_reviewer'] = d['mentioned_reviewer']
                review_specific_discussions.append(discussion_entry)
            
            # Format rebuttals and discussions - store each entry separately
            formatted_rebuttals = format_rebuttals_separated(review_specific_rebuttals)
            formatted_discussions = format_discussions_separated(review_specific_discussions)
            
            # Add new fields: final_decision, rebuttals and reviewer_discussions (separated format)
            row["final_decision"] = final_decision
            row["rebuttals"] = formatted_rebuttals  # Author rebuttals array (including nested replies)
            row["rebuttal_text"] = join_rebuttals(review_specific_rebuttals)  # Keep merged version for compatibility
            row["n_rebuttals"] = len(review_specific_rebuttals)
            row["paper_n_rebuttals"] = len(paper_rebuttals)  # Number of rebuttals for the entire paper
            
            # New: reviewer discussions (including nested replies)
            row["reviewer_discussions"] = formatted_discussions  # Reviewer discussions array
            row["reviewer_discussion_text"] = join_reviewer_discussions(review_specific_discussions)  # Merged discussion text
            row["n_reviewer_discussions"] = len(review_specific_discussions)
            row["paper_n_discussions"] = len(paper_discussions)  # Number of discussions for the entire paper
            
            # New: complete discussion chain (including all author and reviewer replies)
            # row["all_discussions"] = all_related_discussions  # Complete conversation chain
            row["n_all_discussions"] = len(all_related_discussions)

            # Only add reviews with final_decision and rebuttals (filter withdrawn/incomplete papers and reviews without author replies)
            if final_decision is not None and len(formatted_rebuttals) > 0:
                out.append(row)

            # Special diagnosis for the review you provided
            if not debug_printed and (paper_id == DEBUG_FORUM_ID or rid == DEBUG_REVIEW_ID):
                debug_printed = True
                sys.stderr.write("\n=== DEBUG: Found target review ===\n")
                sys.stderr.write(f"forum: {paper_id}\nreview_id: {rid}\n")
                sys.stderr.write(f"content keys: {list(rcontent.keys())}\n")
                sys.stderr.write(f"review_text length: {len(review_text) if review_text else 0}\n")
                sys.stderr.write(f"n_edits: {n_edits}, first_field: {first_name}, last_field: {last_name}\n")
                sys.stderr.write(f"final_decision: {final_decision}\n")
                sys.stderr.write(f"decision_count: {decision_count}, rebuttal_count: {rebuttal_count}, discussion_count: {discussion_count}\n")
                sys.stderr.write(f"n_rebuttals: {len(review_specific_rebuttals)}\n")
                sys.stderr.write(f"paper_n_rebuttals: {len(paper_rebuttals)}\n")
                sys.stderr.write(f"n_reviewer_discussions: {len(review_specific_discussions)}\n")
                sys.stderr.write(f"paper_n_discussions: {len(paper_discussions)}\n")
                sys.stderr.write(f"rebuttals (separated): {len(formatted_rebuttals)} items\n")
                sys.stderr.write(f"structured fields extracted:\n")
                sys.stderr.write(f"  weaknesses_and_questions: {bool(row.get('weaknesses_and_questions'))}\n")
                sys.stderr.write(f"  confidence: {row.get('confidence')}\n")
                sys.stderr.write(f"  final_rating: {row.get('final_rating')}\n")
                sys.stderr.write(f"All paper rebuttals mapping:\n")
                for i, reb in enumerate(paper_rebuttals):
                    sys.stderr.write(f"  Paper Rebuttal {i+1}: ID={reb['rebuttal_id']}, replyto={reb['replyto']}\n")
                sys.stderr.write(f"Review-specific rebuttals for review {rid}:\n")
                for i, reb in enumerate(formatted_rebuttals):
                    sys.stderr.write(f"  Review Rebuttal {i+1}: ID={reb['rebuttal_id']}, length={len(reb['rebuttal_text'])}, replyto={reb['replyto']}\n")
                if first:
                    sys.stderr.write(f"initial_rating_raw: {first[0]}\n")
                if last:
                    sys.stderr.write(f"final_rating_raw: {last[0]}\n")
                if (first is None and last is None) and not rcontent:
                    sys.stderr.write("WARNING: review content is empty/hidden; likely requires login/permissions.\n")
                sys.stderr.flush()

    # Statistics before and after filtering
    total_papers_processed = len(set(row["paper_id"] for row in out)) if out else 0
    total_reviews_before_filter = len([r for sub in subs for r in sub.details.get("replies", []) if is_official_review(r)])
    total_reviews_after_filter = len(out)
    
    # Final result statistics
    total_decisions = sum(1 for row in out if row["final_decision"] is not None)
    total_reviews_with_rebuttals = sum(1 for row in out if row["n_rebuttals"] > 0)
    total_rebuttals_count = sum(row["n_rebuttals"] for row in out)
    reviews_with_multiple_rebuttals = sum(1 for row in out if row["n_rebuttals"] > 1)
    
    # New: reviewer discussions statistics
    total_reviews_with_discussions = sum(1 for row in out if row["n_reviewer_discussions"] > 0)
    total_discussions_count = sum(row["n_reviewer_discussions"] for row in out)
    reviews_with_multiple_discussions = sum(1 for row in out if row["n_reviewer_discussions"] > 1)
    
    print(f"Statistics:")
    print(f"- Total papers processed: {total_papers_processed}")
    print(f"- Total reviews before filtering: {total_reviews_before_filter}")
    print(f"- Total reviews after filtering (with decisions AND rebuttals): {total_reviews_after_filter}")
    print(f"- Reviews filtered out (no decision OR no rebuttals): {total_reviews_before_filter - total_reviews_after_filter}")
    print(f"- Reviews with rebuttals: {total_reviews_with_rebuttals} (100% after filtering)")
    print(f"- Total rebuttals found: {total_rebuttals_count}")
    print(f"- Reviews with multiple rebuttals: {reviews_with_multiple_rebuttals}")
    print(f"- Average rebuttals per review: {total_rebuttals_count/total_reviews_after_filter:.2f}" if total_reviews_after_filter > 0 else "- Average rebuttals per review: 0")
    print(f"- Reviews with reviewer discussions: {total_reviews_with_discussions}")
    print(f"- Total reviewer discussions found: {total_discussions_count}")
    print(f"- Reviews with multiple discussions: {reviews_with_multiple_discussions}")
    print(f"- Average discussions per review: {total_discussions_count/total_reviews_after_filter:.2f}" if total_reviews_after_filter > 0 else "- Average discussions per review: 0")
    
    # New: complete discussion chain statistics
    total_reviews_with_all_discussions = sum(1 for row in out if row["n_all_discussions"] > 0)
    total_all_discussions_count = sum(row["n_all_discussions"] for row in out)
    print(f"- Reviews with any discussions (including nested): {total_reviews_with_all_discussions}")
    print(f"- Total discussions found (including nested): {total_all_discussions_count}")
    print(f"- Average all discussions per review: {total_all_discussions_count/total_reviews_after_filter:.2f}" if total_reviews_after_filter > 0 else "- Average all discussions per review: 0")
    
    # Output JSONL - add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"iclr2024_reviews_with_decision_rebuttal_structured_{timestamp}.jsonl"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(out)} review rows to {output_filename}")

if __name__ == "__main__":
    main()
