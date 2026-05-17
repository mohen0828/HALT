from __future__ import annotations

import re


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"([.!?\n;。！？；]+)", text)
    sentences: list[str] = []
    for idx in range(0, len(parts) - 1, 2):
        chunk = (parts[idx] + parts[idx + 1]).strip()
        if chunk:
            sentences.append(chunk)
    if len(parts) % 2 != 0 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    if not sentences and text.strip():
        sentences.append(text.strip())
    return sentences


def score_input_text(question: str, agent_id: str, sentence: str) -> str:
    return f"Target:\n[{agent_id}]: {sentence}\n\nContext:\n{question}"

