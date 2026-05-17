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
from halt.gsm8k import extract_numeric_answer, is_correct_number
from halt.llm import call_chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect full MAS debate traces for evaluator training.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--router-data", default=str(DEFAULT_DATA_DIR / "router_training_data.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR / "mas_traces.jsonl"))
    parser.add_argument("--all", action="store_true", help="Use all router records instead of needs_mas records only.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    add_common_api_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = load_api_config(args)
    cfg = load_halt_config(args.config)

    candidates = read_jsonl(args.router_data)
    if not args.all:
        candidates = [row for row in candidates if row.get("needs_mas")]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    output_path = Path(args.output)
    processed = {row["global_id"] for row in read_jsonl(output_path)}
    candidates = [row for row in candidates if row["global_id"] not in processed]

    lock = threading.Lock()

    def run_one(item: dict[str, object]) -> dict[str, object] | None:
        context = (
            f"Problem:\n{item['question']}\n\n"
            "Discuss step by step. If you have a final numerical answer, put it after '#### '."
        )
        transcript: list[dict[str, object]] = []
        prompt_tokens = 0
        completion_tokens = 0
        turn = 1

        for _ in range(cfg.debate_rounds):
            for agent in cfg.agents:
                prompt = f"{context}\n\n[Instruction]\n{agent.role_prompt}"
                reply, usage = call_chat_completion(prompt, agent.model, api, temperature=0.0)
                if not reply:
                    return None
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                transcript.append(
                    {
                        "turn": turn,
                        "agent": agent.agent_id,
                        "model": agent.model,
                        "text": reply,
                    }
                )
                context += f"\n\n[{agent.agent_id}]: {reply}"
                turn += 1

        final_prompt = (
            f"{context}\n\n[Instruction]\n"
            "Conclude with the final numerical answer only, and put it after '#### '."
        )
        final_reply, usage = call_chat_completion(final_prompt, cfg.final_model, api, temperature=0.0)
        if not final_reply:
            return None
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)

        final_pred = extract_numeric_answer(final_reply)
        record: dict[str, object] = {
            "global_id": item["global_id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "mas_transcript": transcript,
            "mas_final_response": final_reply,
            "mas_final_pred": final_pred,
            "mas_is_correct": is_correct_number(final_pred, str(item["ground_truth"])),
            "mas_token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        with lock:
            append_jsonl(output_path, record)
            processed.add(str(item["global_id"]))
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(run_one, candidates), total=len(candidates), desc="mas traces"))

    print(f"Saved MAS traces to {output_path}")


if __name__ == "__main__":
    main()
