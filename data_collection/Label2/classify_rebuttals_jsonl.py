import argparse, asyncio, json, sys, time, random, re
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict, Counter
from openai import AsyncOpenAI, OpenAIError
from tqdm.asyncio import tqdm

LABELS = [
    "CRP",  # Concrete Revision Provided
    "SRP",  # Specific Revision Plan
    "VCR",  # Vague Commitment to Revise
    "DWC",  # Defense Without Change
    "DRF",  # Deflection / Reviewer-Faulting
]

# Task definition (compressed into system prompt to save tokens)
TASK_DEFS = """
Return only a compact JSON object: {"impact": "<one_of:[CRP,SRP,VCR,DWC,DRF]>"}.

Categories:
- CRP: Concrete revision already made or concrete, verifiable artifact provided (new text/sections, new experiments/tables/figures, code/data links, numbers).
  Cues: "We added/updated...", "Section X rewritten...", "New ablation in Sec. ... shows ...", "Code/data at ...".
- SRP: Specific revision plan committed, but not yet implemented; where/what to revise is specific.
  Cues: "We will add ablation in Sec. X...", "We will redraw Fig. ...", "We will clarify definitions in §...".
- VCR: Vague promise to revise; no concrete actions, locations, or artifacts.
  Cues: "We will revise accordingly.", "We will improve writing/clarity.".
- DWC: Defend current paper as-is; no new change proposed.
  Cues: "Already covered in Sec. ...", "Setup is standard", "Claim stands".
- DRF: Shift issue to reviewer or avoid underlying point; no change offered.
  Cues: "Reviewer misinterprets ...", "Out of scope ...", "Reviewer phrasing is incorrect".
"""

SYSTEM_PROMPT = (
    "You are a precise, deterministic classifier for rebuttal responses. "
    + TASK_DEFS.strip()
)

def build_user_prompt(text: str) -> str:
    return f"Rebuttal Response:\n{text}\n\nReturn only the JSON."

# ---- Simple keyword fallback (ensures no interruption when API fails) ----
FALLBACK_PATTERNS = [
    ("CRP", [
        r"\bwe (have )?added\b", r"\bwe (have )?updated\b", r"\bwe (have )?revised\b",
        r"\bsection\s*\d", r"\bfig(ure)?\s*\d", r"\btable\s*\d", r"\bablation\b",
        r"\bcode\b", r"\bdata\b", r"https?://", r"\bappendix\b"
    ]),
    ("SRP", [
        r"\bwe will\b", r"\bwe plan to\b", r"\bwe intend to\b", r"\bwe shall\b",
        r"\bwill add\b", r"\bwill revise\b", r"\bwill clarify\b", r"\bwill redraw\b"
    ]),
    ("DRF", [
        r"\bmisinterpret(s|ation)?\b", r"\bout of scope\b", r"\bnot our goal\b",
        r"\bincorrect\b", r"\bmisread\b"
    ]),
    ("DWC", [
        r"\balready (covered|addressed)\b", r"\bas (is|it is)\b", r"\bstandard setup\b",
        r"\bclaim stands\b", r"\bwe believe\b"
    ]),
    ("VCR", [
        r"\bwe will (address|improve|revise)\b", r"\bwe will work on\b",
        r"\bwe will make it better\b"
    ]),
]

def fallback_rule(content: str) -> str:
    text = content.lower()
    for label, pats in FALLBACK_PATTERNS:
        for p in pats:
            if re.search(p, text):
                return label
    # Default to the most conservative label: VCR
    return "VCR"

# ---- Async client with rate limiting ----
class Classifier:
    def __init__(self, api_key: str, organization: str, project: str,
                 model: str, rpm: int, concurrency: int, max_retries: int = 6):
        self.client = AsyncOpenAI(api_key="xxx",
                                  organization="xxx",
                                  project="xxx")
        self.model = model
        self.sema = asyncio.Semaphore(max(1, concurrency))
        self.rpm = max(1, rpm)
        self.min_interval = 60.0 / self.rpm
        self._last_ts = 0.0
        self._lock = asyncio.Lock()
        self.max_retries = max_retries
        
        # Statistics
        self.api_calls = 0
        self.fallback_calls = 0
        self.errors = 0

    async def _throttle(self):
        async with self._lock:
            now = time.time()
            wait = self._last_ts + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.time()

    async def classify(self, content: str) -> tuple[str, bool]:
        # Empty or very short text goes directly to fallback
        if not content or not content.strip():
            self.fallback_calls += 1
            return "VCR", True
        
        delay = 0.5
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.sema:
                    await self._throttle()
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user_prompt(content)}
                        ],
                        response_format={"type": "json_object"},
                        # temperature=0,
                        max_completion_tokens=200  # Increase token limit to ensure complete response
                    )
                # Parse {"impact": "..."}
                # Get response content
                choice = resp.choices[0]
                content_text = choice.message.content
                
                # Check if truncated due to length limit
                if choice.finish_reason == 'length':
                    print(f"⚠️ API response truncated (finish_reason=length), using keyword fallback", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                
                if not content_text or not content_text.strip():
                    print(f"⚠️ API returned empty content (finish_reason={choice.finish_reason}), using keyword fallback", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                
                try:
                    data = json.loads(content_text)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parse failed. API returned: '{content_text[:100]}...'. Using keyword fallback", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                
                label = (data.get("impact") or "").strip()
                self.api_calls += 1
                if label in LABELS:
                    return label, False
                # Returned an unexpected string; try to normalize
                upper = label.upper()
                if upper in LABELS:
                    return upper, False
                # API returned an invalid label; use fallback
                print(f"⚠️ API returned an invalid label: '{label}'. Using keyword fallback", file=sys.stderr)
                self.fallback_calls += 1
                return fallback_rule(content), True
            except (OpenAIError, asyncio.TimeoutError) as e:
                self.errors += 1
                if attempt == self.max_retries:
                    # Final fallback
                    print(f"⚠️ API call failed ({e.__class__.__name__}). Using keyword fallback", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                print(f"⚠️ API call failed, retry {attempt}: {e}", file=sys.stderr)
                await asyncio.sleep(delay + random.uniform(0, 0.25))
                delay *= 2
            except Exception as e:
                self.errors += 1
                print(f"⚠️ Unexpected error: {e}. Using keyword fallback", file=sys.stderr)
                self.fallback_calls += 1
                return fallback_rule(content), True
        
        # Should never reach here, but keep a safe fallback
        self.fallback_calls += 1
        return fallback_rule(content), True

# ---- I/O & main flow ----
async def process_file(in_path: Path, out_path: Path, clf: Classifier):
    # Statistics
    total_labels = Counter()
    perspective_labels = defaultdict(Counter)
    fallback_count = 0
    
    # Pre-read to count total items
    total = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mlist = obj.get("weakness_rebuttal_mappings") or []
            for _ in mlist:
                total += 1

    print(f"📊 Starting classification for {total} rebuttals", file=sys.stderr)
    
    # Main processing
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        # Create progress bar
        pbar = tqdm(total=total, desc="Progress", file=sys.stderr, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for line in fin:
            if not line.strip():
                continue
            try:
                paper = json.loads(line)
            except Exception as e:
                print(f"⚠️ Failed to parse JSON: {e}", file=sys.stderr)
                continue

            mappings = paper.get("weakness_rebuttal_mappings")
            if isinstance(mappings, list):
                # Classify all rebuttals in this paper concurrently
                tasks = []
                idxs = []
                for i, m in enumerate(mappings):
                    rr = (m or {}).get("rebuttal_response") or {}
                    content = rr.get("content", "")
                    # Async task: get impact label
                    tasks.append(asyncio.create_task(clf.classify(content)))
                    idxs.append(i)

                if tasks:
                    results = await asyncio.gather(*tasks)
                    for i, (impact, is_fallback) in zip(idxs, results):
                        # Write impact back into rebuttal_response
                        if isinstance(mappings[i], dict):
                            rr = mappings[i].get("rebuttal_response")
                            if isinstance(rr, dict):
                                rr["impact"] = impact
                            
                            # Statistics
                            total_labels[impact] += 1
                            if is_fallback:
                                fallback_count += 1
                            
                            # Get perspective and update stats
                            wp = mappings[i].get("weakness_point") or {}
                            perspective = wp.get("perspective", "Unknown")
                            perspective_labels[perspective][impact] += 1
                        
                        pbar.update(1)

            # Write this line (paper object)
            fout.write(json.dumps(paper, ensure_ascii=False) + "\n")
        
        pbar.close()
    
    # Print summary statistics
    print("\n" + "="*60, file=sys.stderr)
    print("📈 Classification complete. Summary:", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    print(f"🔧 API calls: {clf.api_calls}", file=sys.stderr)
    print(f"⚠️ Fallback used: {clf.fallback_calls} ({clf.fallback_calls/total*100:.1f}%)", file=sys.stderr)
    print(f"❌ Errors: {clf.errors}", file=sys.stderr)
    
    print("\n📊 Overall label distribution:", file=sys.stderr)
    for label in LABELS:
        count = total_labels[label]
        print(f"  {label}: {count} ({count/total*100:.1f}%)", file=sys.stderr)
    
    print("\n📊 Label distribution by perspective:", file=sys.stderr)
    for perspective in sorted(perspective_labels.keys()):
        if perspective == "Unknown":
            continue
        total_per_perspective = sum(perspective_labels[perspective].values())
        print(f"\n  {perspective} (total: {total_per_perspective}):", file=sys.stderr)
        for label in LABELS:
            count = perspective_labels[perspective][label]
            if count > 0:
                print(f"    {label}: {count} ({count/total_per_perspective*100:.1f}%)", file=sys.stderr)
    
    if perspective_labels["Unknown"]:
        total_unknown = sum(perspective_labels["Unknown"].values())
        print(f"\n  Unknown Perspective (total: {total_unknown}):", file=sys.stderr)
        for label in LABELS:
            count = perspective_labels["Unknown"][label]
            if count > 0:
                print(f"    {label}: {count} ({count/total_unknown*100:.1f}%)", file=sys.stderr)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, type=str)
    ap.add_argument("--out_path", required=True, type=str)
    ap.add_argument("--api_key", type=str)
    ap.add_argument("--organization", default=None, type=str)
    ap.add_argument("--project", default=None, type=str)
    ap.add_argument("--model", default="gpt-5-mini", type=str)
    ap.add_argument("--rpm", default=1000, type=int, help="Requests per minute limit")
    ap.add_argument("--concurrency", default=10, type=int, help="Concurrent tasks")
    ap.add_argument("--max_retries", default=6, type=int)
    return ap.parse_args()

async def amain():
    args = parse_args()
    clf = Classifier(
        api_key=args.api_key,
        organization=args.organization,
        project=args.project,
        model=args.model,
        rpm=args.rpm,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )
    await process_file(Path(args.in_path), Path(args.out_path), clf)

if __name__ == "__main__":
    # Windows event loop compatibility (if needed)
    # import platform, asyncio
    # if platform.system().lower().startswith("win"):
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(amain())