
import asyncio
import random
from typing import Callable, Dict, List, Sequence, Type, Optional, Union

import os
import dotenv
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel
import pydantic_core
import backoff
from tqdm.asyncio import tqdm_asyncio

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI (non-Azure) async client
# ---------------------------------------------------------------------------
_ORG = "xxx"
_PROJECT = "xxx"
_API_KEY = "xxx"

if not _API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Please set it in your environment or a .env file.")

openai_client = AsyncOpenAI(
    api_key=_API_KEY,
    organization=_ORG if _ORG else None,
    project=_PROJECT if _PROJECT else None,
)

client_dict: Dict[str, List[openai.AsyncClient]] = {
    "gpt-5-chat": [openai_client],
    "gpt-5-mini": [openai_client],
    "gpt-4.1": [openai_client],
    "gpt-4.1-mini": [openai_client],
    "o4-mini": [openai_client],
}

__all__: Sequence[str] = (
    "process_single_example",
    "process_single_example_unstructured",
    "process_data",
    "aclose_all_clients",
    "client_dict",
)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
async def _call_llm(
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    model: str,
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Optional[BaseModel]:
    """Structured helper that selects a random client for *model* and invokes
    Chat Completions with Pydantic parsing (beta endpoint)."""
    try:
        client = random.choice(client_dict[model])
        # Prefer parse() for structured output
        resp = await client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            # Some SDKs expect max_output_tokens here; others expect max_completion_tokens.
            # We pass both for compatibility; the SDK will ignore unknown kwargs.
            max_output_tokens=max_tokens,
            max_completion_tokens=max_tokens,
            response_format=response_model,
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        print(f"❌ Error in _call_llm: {type(e).__name__}: {e}")
        return None

async def _call_llm_unstructured(
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> str:
    """Unstructured helper that invokes Chat Completions and returns raw text."""
    client = random.choice(client_dict[model])
    # Use the most compatible arg name `max_tokens` for chat.completions
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Robustly extract text content
    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""
    return text

# ---------------------------------------------------------------------------
# Back-off decorator for transient errors
# ---------------------------------------------------------------------------
def _backoff_decorator():
    return backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIError),
        max_tries=8,
    )

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
@_backoff_decorator()
async def process_single_example(
    example: Dict,
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    build_messages: Callable[[Dict], List[Dict[str, str]]],
    response_model: Type[BaseModel],
    model: str,
    post_process: Optional[Callable[[Dict, BaseModel, str], Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Optional[Dict]:
    for attempt in range(3):
        try:
            parsed = await _call_llm(
                client_dict=client_dict,
                model=model,
                messages=build_messages(example),
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if parsed is not None:
                if post_process is not None:
                    return post_process(example, parsed, model)
                example.update(parsed.model_dump())
                return example
        except pydantic_core._pydantic_core.ValidationError as ve:
            if attempt == 2:
                print(f"ValidationError (final) for example: {ve}")
                return None
            continue
    return None

@_backoff_decorator()
async def process_single_example_unstructured(
    example: Dict,
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    build_messages: Callable[[Dict], List[Dict[str, str]]],
    model: str,
    post_process: Optional[Callable[[Dict, Union[str, BaseModel], str], Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Optional[Dict]:
    raw_text = await _call_llm_unstructured(
        client_dict=client_dict,
        model=model,
        messages=build_messages(example),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if post_process is not None:
        return post_process(example, raw_text, model)
    example["response"] = raw_text
    return example

async def process_data(
    data: List[Dict],
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    build_messages: Callable[[Dict], List[Dict[str, str]]],
    response_model: Type[BaseModel],
    models: List[str],
    enable_structured_output: bool = True,
    post_process: Optional[Callable[[Dict, Union[BaseModel, str], str], Dict]] = None,
    max_concurrency: int = 32,
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> List[Optional[Dict]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _worker(ex: Dict) -> Optional[Dict]:
        async with semaphore:
            model_choice = random.choice(models)
            if enable_structured_output:
                return await process_single_example(
                    ex,
                    client_dict=client_dict,
                    build_messages=build_messages,
                    response_model=response_model,
                    model=model_choice,
                    post_process=post_process,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                return await process_single_example_unstructured(
                    ex,
                    client_dict=client_dict,
                    build_messages=build_messages,
                    model=model_choice,
                    post_process=post_process,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

    tasks = [asyncio.create_task(_worker(ex)) for ex in data]
    return await tqdm_asyncio.gather(*tasks, desc="Processing Items", total=len(tasks))

async def aclose_all_clients() -> None:
    async def _try_close(client: openai.AsyncClient) -> None:
        try:
            res = client.aclose()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass

    tasks = []
    seen = set()
    for lst in client_dict.values():
        for c in lst:
            if id(c) in seen:
                continue
            seen.add(id(c))
            tasks.append(asyncio.create_task(_try_close(c)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
