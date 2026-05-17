from __future__ import annotations

import json
import re
from pathlib import Path


def extract_gsm8k_gold(answer_text: str) -> str:
    if "####" in answer_text:
        return normalize_number(answer_text.split("####")[-1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_text.replace(",", ""))
    return numbers[-1] if numbers else answer_text.strip()


def normalize_number(text: str) -> str:
    text = text.replace(",", "").strip()
    text = re.sub(r"[^\d.\-]", "", text)
    return text


def extract_numeric_answer(text: str) -> str:
    if "####" in text:
        ans = normalize_number(text.split("####")[-1])
        if ans:
            return ans

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        ans = normalize_number(boxed_match.group(1))
        if ans:
            return ans

    answer_matches = re.findall(r"(?:answer|therefore|so)[^\d\-]*(-?\d+(?:\.\d+)?)", text, flags=re.I)
    if answer_matches:
        return answer_matches[-1]

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1]

    return "Z"


def is_correct_number(pred: str, gold: str, tolerance: float = 1e-5) -> bool:
    if pred == "Z":
        return False
    try:
        return abs(float(pred) - float(gold)) <= tolerance
    except ValueError:
        return pred.strip() == gold.strip()


def load_gsm8k_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "global_id": item.get("global_id", f"gsm8k_{idx}"),
                    "question": item["question"],
                    "ground_truth": extract_gsm8k_gold(item["answer"]),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def gsm8k_single_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "[Instruction]\n"
        "Reason step by step. Put the final numerical answer after '#### ' at the end, "
        "for example '#### 42'."
    )

