"""
Thin client for the local Ollama API.

Provides:
  - chat()         — single-turn completion (returns str)
  - extract_json() — completion that parses and returns a dict
  - score_links()  — score a list of URLs for relevance to a query (returns list[float])
  - clean_text()   — normalise / clean raw scraped text
  - extract_listings() — intelligently pull structured fields from raw HTML/text
"""

from __future__ import annotations

import json
import re
import time
import logging
from typing import Any, Optional

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.expanduser("~"))
del _sys, _os

import requests
from user_profile import get_profile
from tool_registry import run_with_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"

# Fast extraction model for scraping; fall back to reasoning model
PRIMARY_MODEL   = "qwen3:1.7b"
FALLBACK_MODEL  = "deepseek-r1:7b"

_active_model: Optional[str] = None   # resolved lazily


def _resolve_model() -> str:
    """Return the first available model from the running Ollama instance."""
    global _active_model
    if _active_model:
        return _active_model
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.ok:
            names = {m["name"] for m in resp.json().get("models", [])}
            for candidate in (PRIMARY_MODEL, FALLBACK_MODEL):
                if candidate in names:
                    _active_model = candidate
                    logger.info("Ollama: using model %s", candidate)
                    return candidate
            # Any model will do
            if names:
                _active_model = next(iter(names))
                logger.warning("Ollama: preferred models not found; using %s", _active_model)
                return _active_model
    except Exception as exc:
        logger.warning("Ollama: could not reach %s — %s", OLLAMA_BASE, exc)
    _active_model = FALLBACK_MODEL
    return _active_model


# ---------------------------------------------------------------------------
# Core chat helper
# ---------------------------------------------------------------------------

def chat(
    prompt: str,
    *,
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 60,
    max_retries: int = 2,
) -> str:
    """Send a prompt to Ollama and return the response text.

    Returns empty string on any failure so callers can degrade gracefully.
    """
    model = model or _resolve_model()

    # Prepend user personalization context to every system prompt
    profile_context = get_profile().system_context("web-scraper")
    if profile_context:
        system = f"{profile_context}\n\n{system}" if system else profile_context

    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [],
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json=payload,
                timeout=timeout,
            )
            if resp.ok:
                return resp.json().get("message", {}).get("content", "").strip()
            logger.warning("Ollama %s attempt %d: HTTP %s", model, attempt, resp.status_code)
        except requests.Timeout:
            logger.warning("Ollama %s attempt %d: timeout", model, attempt)
        except Exception as exc:
            logger.warning("Ollama %s attempt %d: %s", model, attempt, exc)
        if attempt < max_retries:
            time.sleep(2 ** attempt)

    return ""


# ---------------------------------------------------------------------------
# JSON extraction wrapper
# ---------------------------------------------------------------------------

def extract_json(prompt: str, *, system: str = "", **kwargs) -> dict | list | None:
    """Run a chat completion and parse the first JSON object/array from the reply."""
    raw = chat(prompt, system=system, **kwargs)
    if not raw:
        return None

    # Strip ```json ... ``` fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    # Try to locate and parse the first {...} or [...] block
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

    return None


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def score_links(urls: list[str], query: str, *, context: str = "") -> list[float]:
    """
    Score each URL for relevance to *query* on a 0–1 scale.

    Returns a list of floats (same length as *urls*) so the caller can
    rank / filter links before fetching them.
    Gracefully returns 0.5 for all links if the LLM is unavailable.
    """
    if not urls:
        return []

    # Build a compact numbered list so the LLM response is easy to parse
    numbered = "\n".join(f"{i+1}. {u}" for i, u in enumerate(urls))
    ctx_note = f"\nPage context: {context[:400]}" if context else ""

    prompt = (
        f"You are a web-scraping assistant. The user is searching for: \"{query}\".{ctx_note}\n\n"
        f"Below is a numbered list of URLs found on a page.\n"
        f"Rate each URL from 0.0 (completely irrelevant) to 1.0 (highly relevant) "
        f"based on how likely following it will yield results related to the query.\n"
        f"Reply ONLY with a JSON array of numbers in the same order as the list. "
        f"No explanation.\n\n"
        f"{numbered}"
    )

    result = extract_json(prompt, timeout=45)
    if isinstance(result, list) and len(result) == len(urls):
        try:
            return [max(0.0, min(1.0, float(x))) for x in result]
        except (TypeError, ValueError):
            pass

    logger.warning("score_links: LLM returned unexpected result; defaulting to 0.5")
    return [0.5] * len(urls)


def clean_text(raw: str, *, hint: str = "") -> str:
    """
    Use the LLM to clean and normalise raw scraped text.

    hint — optional sentence describing what the text is about.
    Returns the cleaned text, or *raw* unchanged on LLM failure.
    """
    if not raw or len(raw) < 20:
        return raw

    hint_note = f" The text is about: {hint}." if hint else ""
    prompt = (
        f"Clean the following raw scraped web text.{hint_note}\n"
        "Remove HTML artifacts, boilerplate navigation text, cookie notices, "
        "repeated whitespace, and unrelated junk. "
        "Return only the cleaned, readable content. "
        "Keep all product/price/seller details intact.\n\n"
        f"---\n{raw[:3000]}\n---"
    )
    cleaned = chat(prompt, temperature=0.1, timeout=60)
    return cleaned if cleaned else raw


def extract_listings(html_or_text: str, query: str, site: str = "") -> list[dict]:
    """
    Ask the LLM to extract structured product listings from raw HTML or text.

    Uses run_with_tools so the model can perform additional web searches during
    extraction (e.g. to resolve ambiguous prices or verify listings).

    Returns a list of dicts with keys: title, price, url, condition, shipping,
    seller, location.  Returns [] on LLM failure.
    """
    # Truncate to keep the prompt manageable
    content = html_or_text[:6000]
    site_note = f" from {site}" if site else ""

    profile_context = get_profile().system_context("web-scraper")
    system_content = (
        "You are a precise data-extraction assistant. "
        "Extract structured product listings from web content. "
        "Always respond with valid JSON only. "
        "You may use web_search or fetch_url tools if you need additional context."
    )
    if profile_context:
        system_content = f"{profile_context}\n\n{system_content}"

    user_prompt = (
        f"Extract product listings{site_note} related to: \"{query}\".\n\n"
        "From the content below, extract ALL product listings you can find. "
        "For each listing output a JSON object with these fields:\n"
        "  title (string), price (string, e.g. '$12.99'), url (string or null),\n"
        "  condition (string or null), shipping (string or null),\n"
        "  seller (string or null), location (string or null)\n\n"
        "Reply with a JSON array of these objects ONLY. No explanation.\n\n"
        f"---CONTENT---\n{content}\n---END---"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    try:
        model = _resolve_model()
        raw, _ = run_with_tools(messages, model, ollama_url=OLLAMA_BASE, max_rounds=4)
    except Exception as exc:
        logger.warning("run_with_tools failed in extract_listings: %s", exc)
        raw = chat(user_prompt, system=system_content, timeout=90)

    if not raw:
        return []

    # Parse JSON out of the response
    raw_clean = re.sub(r"```(?:json)?\s*", "", raw)
    raw_clean = re.sub(r"```\s*$", "", raw_clean)
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, raw_clean, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list):
                    return [r for r in result if isinstance(r, dict) and r.get("title")]
                if isinstance(result, dict) and result.get("title"):
                    return [result]
            except json.JSONDecodeError:
                pass
    return []


def is_ollama_available() -> bool:
    """Quick ping to check if Ollama is reachable."""
    try:
        return requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3).ok
    except Exception:
        return False
