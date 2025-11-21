import argparse, asyncio, json, sys, time, random, re
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict, Counter
from openai import AsyncOpenAI, OpenAIError
from tqdm.asyncio import tqdm

# D:\anaconda3\envs\literature\python.exe classify_rebuttals_jsonl.py --in_path rebuttals.jsonl --out_path rebuttals_labeled.jsonl --model gpt-5-mini --rpm 2000 --concurrency 10

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
    # Default most conservative: VCR
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
                    print(f"⚠️ API response truncated (finish_reason=length), using keyword fallback rules", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                
                if not content_text or not content_text.strip():
                    print(f"⚠️ API returned empty content (finish_reason={choice.finish_reason}), using keyword fallback rules", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                
                try:
                    data = json.loads(content_text)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析失败，API返回内容: '{content_text[:100]}...'，使用关键词兜底规则", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                
                label = (data.get("impact") or "").strip()
                self.api_calls += 1
                if label in LABELS:
                    return label, False
                # 返回了非预期字符串，尝试修正
                upper = label.upper()
                if upper in LABELS:
                    return upper, False
                # API返回了无效标签，使用兜底规则
                print(f"⚠️ API返回了无效标签: '{label}'，使用关键词兜底规则", file=sys.stderr)
                self.fallback_calls += 1
                return fallback_rule(content), True
            except (OpenAIError, asyncio.TimeoutError) as e:
                self.errors += 1
                if attempt == self.max_retries:
                    # 最终兜底
                    print(f"⚠️ API调用失败（{e.__class__.__name__}），使用关键词兜底规则", file=sys.stderr)
                    self.fallback_calls += 1
                    return fallback_rule(content), True
                print(f"⚠️ API调用失败，第{attempt}次重试: {e}", file=sys.stderr)
                await asyncio.sleep(delay + random.uniform(0, 0.25))
                delay *= 2
            except Exception as e:
                self.errors += 1
                print(f"⚠️ 未知错误: {e}，使用关键词兜底规则", file=sys.stderr)
                self.fallback_calls += 1
                return fallback_rule(content), True
        
        # 不应该到达这里，但为了安全
        self.fallback_calls += 1
        return fallback_rule(content), True

# ---- I/O & 主流程 ----
async def process_file(in_path: Path, out_path: Path, clf: Classifier):
    # 统计信息
    total_labels = Counter()
    perspective_labels = defaultdict(Counter)
    fallback_count = 0
    
    # 预读取计算总数
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

    print(f"📊 开始处理，共{total}个rebuttal需要分类", file=sys.stderr)
    
    # 正式处理
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        # 创建进度条
        pbar = tqdm(total=total, desc="分类进度", file=sys.stderr, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for line in fin:
            if not line.strip():
                continue
            try:
                paper = json.loads(line)
            except Exception as e:
                print(f"⚠️ 解析JSON失败: {e}", file=sys.stderr)
                continue

            mappings = paper.get("weakness_rebuttal_mappings")
            if isinstance(mappings, list):
                # 针对该 paper 内所有 rebuttal 并发分类
                tasks = []
                idxs = []
                for i, m in enumerate(mappings):
                    rr = (m or {}).get("rebuttal_response") or {}
                    content = rr.get("content", "")
                    # 异步任务：获取 impact
                    tasks.append(asyncio.create_task(clf.classify(content)))
                    idxs.append(i)

                if tasks:
                    results = await asyncio.gather(*tasks)
                    for i, (impact, is_fallback) in zip(idxs, results):
                        # 写回到 rebuttal_response 中新增 impact
                        if isinstance(mappings[i], dict):
                            rr = mappings[i].get("rebuttal_response")
                            if isinstance(rr, dict):
                                rr["impact"] = impact
                            
                            # 统计信息
                            total_labels[impact] += 1
                            if is_fallback:
                                fallback_count += 1
                            
                            # 获取perspective并统计
                            wp = mappings[i].get("weakness_point") or {}
                            perspective = wp.get("perspective", "Unknown")
                            perspective_labels[perspective][impact] += 1
                        
                        pbar.update(1)

            # 写出本行（paper 对象）
            fout.write(json.dumps(paper, ensure_ascii=False) + "\n")
        
        pbar.close()
    
    # 输出统计结果
    print("\n" + "="*60, file=sys.stderr)
    print("📈 分类完成！统计结果：", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    print(f"🔧 API调用次数: {clf.api_calls}", file=sys.stderr)
    print(f"⚠️ 兜底规则使用次数: {clf.fallback_calls} ({clf.fallback_calls/total*100:.1f}%)", file=sys.stderr)
    print(f"❌ 错误次数: {clf.errors}", file=sys.stderr)
    
    print("\n📊 总体Label分布:", file=sys.stderr)
    for label in LABELS:
        count = total_labels[label]
        print(f"  {label}: {count} ({count/total*100:.1f}%)", file=sys.stderr)
    
    print("\n📊 各Perspective的Label分布:", file=sys.stderr)
    for perspective in sorted(perspective_labels.keys()):
        if perspective == "Unknown":
            continue
        total_per_perspective = sum(perspective_labels[perspective].values())
        print(f"\n  {perspective} (总数: {total_per_perspective}):", file=sys.stderr)
        for label in LABELS:
            count = perspective_labels[perspective][label]
            if count > 0:
                print(f"    {label}: {count} ({count/total_per_perspective*100:.1f}%)", file=sys.stderr)
    
    if perspective_labels["Unknown"]:
        total_unknown = sum(perspective_labels["Unknown"].values())
        print(f"\n  Unknown Perspective (总数: {total_unknown}):", file=sys.stderr)
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
    # Windows 事件循环兼容（如需）
    # import platform, asyncio
    # if platform.system().lower().startswith("win"):
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(amain())
