from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halt.config import (
    DEFAULT_DATA_DIR,
    add_common_api_args,
    append_jsonl,
    load_api_config,
    load_halt_config,
    read_jsonl,
)
from halt.llm import call_chat_completion
from halt.text import split_into_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MAS trace sentences with an oracle LLM.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=str(DEFAULT_DATA_DIR / "mas_traces.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR / "scored_traces.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    add_common_api_args(parser)
    return parser.parse_args()


def build_scoring_prompt(item: dict[str, object], segmented: str) -> str:
    return f"""You are an impartial logic arbiter with access to the ground-truth answer.

Evaluate each segmented sentence in the multi-agent trace for this GSM8K problem.
Assign exactly one score from [0.0, 0.25, 0.5, 0.75, 1.0]:
- 1.0: crucial correct reasoning, decisive calculation, or correction of a major error.
- 0.75: helpful correct intermediate reasoning.
- 0.5: neutral transition, restatement, or harmless filler.
- 0.25: distracting or irrelevant content.
- 0.0: incorrect calculation, false assumption, or reasoning that contradicts the ground truth.

Problem:
{item["question"]}

Ground truth numerical answer:
{item["ground_truth"]}

Segmented trace:
{segmented}

Return only a JSON object with this schema:
{{
  "evaluations": [
    {{"sentence_id": "T1_S0", "score": 0.5}}
  ]
}}
Evaluate every sentence id exactly once.
"""


def segment_trace(item: dict[str, object]) -> tuple[str, dict[str, str]]:
    transcript = item.get("mas_transcript", [])
    sentence_mapping: dict[str, str] = {}
    blocks: list[str] = []
    for turn_data in transcript:
        turn = int(turn_data["turn"])
        agent = str(turn_data["agent"])
        text = str(turn_data["text"])
        blocks.append(f"--- [Turn {turn}] {agent} ---")
        for sent_idx, sentence in enumerate(split_into_sentences(text)):
            sid = f"T{turn}_S{sent_idx}"
            sentence_mapping[sid] = sentence
            blocks.append(f"[{sid}]: {sentence}")
        blocks.append("")
    return "\n".join(blocks), sentence_mapping


def parse_scores(raw: str) -> list[dict[str, float | str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise
        data = json.loads(raw[start : end + 1])

    out: list[dict[str, float | str]] = []
    for row in data.get("evaluations", []):
        sid = row.get("sentence_id") or row.get("id") or row.get("sentenceId")
        if not sid:
            continue
        try:
            score = float(row.get("score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        score = min(1.0, max(0.0, score))
        out.append({"sentence_id": sid, "score": score})
    return out


def main() -> None:
    args = parse_args()
    api = load_api_config(args)
    cfg = load_halt_config(args.config)

    inputs = read_jsonl(args.input)
    if args.limit is not None:
        inputs = inputs[: args.limit]
    output_path = Path(args.output)
    processed = {row["global_id"] for row in read_jsonl(output_path)}
    inputs = [row for row in inputs if row["global_id"] not in processed]
    lock = threading.Lock()

    def run_one(item: dict[str, object]) -> dict[str, object] | None:
        segmented, sentence_mapping = segment_trace(item)
        if not sentence_mapping:
            return None
        prompt = build_scoring_prompt(item, segmented)
        reply, _ = call_chat_completion(
            prompt,
            cfg.oracle_model,
            api,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        if not reply:
            return None
        try:
            scores = parse_scores(reply)
        except json.JSONDecodeError:
            return None
        record = dict(item)
        record["fine_grained_scores"] = scores
        record["sentence_mapping"] = sentence_mapping
        with lock:
            append_jsonl(output_path, record)
            processed.add(str(item["global_id"]))
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(run_one, inputs), total=len(inputs), desc="oracle scoring"))

    print(f"Saved scored traces to {output_path}")


if __name__ == "__main__":
    main()
