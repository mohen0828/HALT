from __future__ import annotations

import argparse
import concurrent.futures
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
from halt.gsm8k import extract_numeric_answer, gsm8k_single_prompt, is_correct_number, load_gsm8k_jsonl
from halt.llm import call_chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect HALT router training labels on GSM8K.")
    parser.add_argument("--config", default=None, help="Path to HALT JSON config.")
    parser.add_argument("--gsm8k", required=True, help="GSM8K jsonl file with question/answer fields.")
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR / "router_training_data.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    add_common_api_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = load_api_config(args)
    cfg = load_halt_config(args.config)

    output_path = Path(args.output)
    processed = {row["global_id"] for row in read_jsonl(output_path)}
    examples = [row for row in load_gsm8k_jsonl(args.gsm8k, args.limit) if row["global_id"] not in processed]

    lock = threading.Lock()

    def run_one(item: dict[str, str]) -> dict[str, object] | None:
        prompt = gsm8k_single_prompt(item["question"])
        reply, usage = call_chat_completion(prompt, cfg.single_model, api, temperature=0.0)
        if not reply:
            return None
        pred = extract_numeric_answer(reply)
        is_correct = is_correct_number(pred, item["ground_truth"])
        record: dict[str, object] = {
            "global_id": item["global_id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "single_model": cfg.single_model,
            "single_agent_response": reply,
            "single_agent_pred": pred,
            "is_correct": is_correct,
            "needs_mas": not is_correct,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        with lock:
            append_jsonl(output_path, record)
            processed.add(item["global_id"])
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(run_one, examples), total=len(examples), desc="router labels"))

    print(f"Saved router labels to {output_path}")


if __name__ == "__main__":
    main()
