from __future__ import annotations

import time
from typing import Any

import requests

from .config import ApiConfig


EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def call_chat_completion(
    prompt: str,
    model: str,
    api: ApiConfig,
    temperature: float = 0.0,
    response_format: dict[str, str] | None = None,
) -> tuple[str, dict[str, int]]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api.api_key}",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    for attempt in range(api.max_retries):
        try:
            response = requests.post(api.api_url, json=payload, headers=headers, timeout=api.timeout)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", EMPTY_USAGE) or EMPTY_USAGE
            return content, {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
        except Exception:
            if attempt == api.max_retries - 1:
                return "", EMPTY_USAGE.copy()
            time.sleep(2**attempt)

    return "", EMPTY_USAGE.copy()

