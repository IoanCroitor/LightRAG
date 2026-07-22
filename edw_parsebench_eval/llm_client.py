"""OpenAI-compatible LLM client for ground-truth generation and judging.

Talks to the GPUStack endpoint configured in :mod:`config` (the same one the
LightRAG server uses for generation). The client is intentionally thin: it
wraps ``openai.OpenAI`` and adds (a) a tolerant JSON extractor so we can parse
model output even when it is wrapped in markdown code fences, and (b) simple
retry handling for transient errors.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError

import config


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str):
    """Parse a JSON value from arbitrary model output.

    Tries, in order:
      1. the raw string as JSON,
      2. the contents of a ```json ... ``` / ``` ... ``` fence,
      3. the first balanced-looking ``{...}`` / ``[...]`` substring.
    Raises ``ValueError`` if nothing parseable is found.
    """
    if text is None:
        raise ValueError("empty model output")
    candidate = text.strip()
    attempts = [candidate]
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        attempts.append(fence.group(1))
    # Fall back to first { or [ and the matching last bracket.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            attempts.append(candidate[start : end + 1])
    last_err: Optional[Exception] = None
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except Exception as exc:  # noqa: BLE001 - we try several strategies
            last_err = exc
    raise ValueError(f"could not parse JSON from model output: {last_err}")


def chat(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    """Single-turn chat completion. Retries on transient API errors."""
    client = _get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            kwargs: dict = {
                "model": model or config.LLM_MODEL,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            # Qwen3 thinking consumes the token budget and can truncate JSON.
            if not config.ENABLE_THINKING:
                kwargs["extra_body"] = {"enable_thinking": False}
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            return content
        except (RateLimitError, APIError) as exc:
            last_err = exc
            if attempt == config.MAX_RETRIES:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"LLM call failed after {config.MAX_RETRIES} attempts: {last_err}")


def chat_json(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
):
    """``chat`` that parses the response as JSON via :func:`extract_json`."""
    return extract_json(chat(system, user, model=model, temperature=temperature, max_tokens=max_tokens))
