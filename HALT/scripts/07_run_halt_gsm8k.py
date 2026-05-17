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
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_OUTPUT_DIR,
    add_common_api_args,
    append_jsonl,
    load_api_config,
    load_halt_config,
    read_jsonl,
)
from halt.gsm8k import extract_numeric_answer, gsm8k_single_prompt, is_correct_number, load_gsm8k_jsonl
from halt.llm import call_chat_completion
from halt.text import split_into_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HALT end-to-end on GSM8K.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--gsm8k", required=True)
    parser.add_argument("--router", default=str(DEFAULT_ARTIFACT_DIR / "router_gsm8k.pkl"))
    parser.add_argument("--evaluator", default=str(DEFAULT_ARTIFACT_DIR / "evaluator_deberta_v3"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gsm8k_halt_results.jsonl"))
    parser.add_argument("--summary", default=str(DEFAULT_OUTPUT_DIR / "gsm8k_halt_summary.txt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--architecture", choices=["linear", "fullmesh", "star", "hierarchical"], default=None)
    parser.add_argument("--router-threshold", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_common_api_args(parser)
    return parser.parse_args()


def add_usage(metrics: dict[str, object], usage: dict[str, int]) -> None:
    metrics["prompt_tokens"] = int(metrics["prompt_tokens"]) + usage.get("prompt_tokens", 0)
    metrics["completion_tokens"] = int(metrics["completion_tokens"]) + usage.get("completion_tokens", 0)
    metrics["total_tokens"] = int(metrics["total_tokens"]) + usage.get("total_tokens", 0)


def render_memory(memory: dict[str, str], agent_order: list[str]) -> str:
    blocks = []
    for agent_id in agent_order:
        text = memory.get(agent_id, "").strip()
        if text:
            blocks.append(f"[{agent_id}]:\n{text}")
    return "\n\n".join(blocks)


def build_agent_prompt(question: str, role_prompt: str, debate_history: str) -> str:
    if debate_history.strip():
        return (
            f"{question}\n\n[Debate History]\n{debate_history.strip()}\n\n[Instruction]\n"
            f"{role_prompt}\nIf you have a final numerical answer, put it after '#### '."
        )
    return (
        f"{question}\n\n[Instruction]\n{role_prompt}\n"
        "If you have a final numerical answer, put it after '#### '."
    )


def run_agent_turn(
    question: str,
    agent,
    debate_history: str,
    metrics: dict[str, object],
    cfg,
    api,
    evaluator: SentenceEvaluator,
    evaluator_lock: threading.Lock,
) -> dict[str, object] | None:
    prompt = build_agent_prompt(question, agent.role_prompt, debate_history)
    reply, usage = call_chat_completion(prompt, agent.model, api, temperature=0.0)
    if not reply:
        return None

    add_usage(metrics, usage)
    metrics["api_calls"] = int(metrics["api_calls"]) + 1

    sentences = split_into_sentences(reply)
    if not sentences:
        return {"early_halt": False, "final_ans": "Z", "valid_sentences": [], "reply": reply}

    with evaluator_lock:
        scores = evaluator.score(question, agent.agent_id, sentences)

    extracted = extract_numeric_answer(reply)
    if (
        extracted != "Z"
        and scores
        and max(scores) >= cfg.early_halt_max_threshold
        and min(scores) >= cfg.early_halt_min_threshold
    ):
        return {"early_halt": True, "final_ans": extracted, "valid_sentences": [], "reply": reply}

    valid_sentences = [sentence for sentence, score in zip(sentences, scores) if score >= cfg.prune_keep_threshold]
    return {"early_halt": False, "final_ans": "Z", "valid_sentences": valid_sentences, "reply": reply}


def update_memory(memory: dict[str, str], agent_id: str, sentences: list[str]) -> None:
    if not sentences:
        return
    text = " ".join(sentences).strip()
    old = memory.get(agent_id, "").strip()
    memory[agent_id] = f"{old} {text}".strip() if old else text


def run_mas_linear(question: str, metrics: dict[str, object], cfg, api, evaluator, evaluator_lock):
    agent_ids = [agent.agent_id for agent in cfg.agents]
    memory = {agent_id: "" for agent_id in agent_ids}
    for _ in range(cfg.debate_rounds):
        for idx, agent in enumerate(cfg.agents):
            history = ""
            if idx > 0:
                prev_id = cfg.agents[idx - 1].agent_id
                prev_text = memory.get(prev_id, "").strip()
                if prev_text:
                    history = f"[{prev_id}]:\n{prev_text}"
            turn = run_agent_turn(question, agent, history, metrics, cfg, api, evaluator, evaluator_lock)
            if turn is None:
                continue
            if turn["early_halt"]:
                return str(turn["final_ans"]), True, render_memory(memory, agent_ids)
            update_memory(memory, agent.agent_id, turn["valid_sentences"])
    return "Z", False, render_memory(memory, agent_ids)


def run_mas_fullmesh(question: str, metrics: dict[str, object], cfg, api, evaluator, evaluator_lock):
    agent_ids = [agent.agent_id for agent in cfg.agents]
    memory = {agent_id: "" for agent_id in agent_ids}
    for _ in range(cfg.debate_rounds):
        for agent in cfg.agents:
            peers = [aid for aid in agent_ids if aid != agent.agent_id]
            history = render_memory(memory, peers)
            turn = run_agent_turn(question, agent, history, metrics, cfg, api, evaluator, evaluator_lock)
            if turn is None:
                continue
            if turn["early_halt"]:
                return str(turn["final_ans"]), True, render_memory(memory, agent_ids)
            update_memory(memory, agent.agent_id, turn["valid_sentences"])
    return "Z", False, render_memory(memory, agent_ids)


def run_mas_star(question: str, metrics: dict[str, object], cfg, api, evaluator, evaluator_lock):
    if len(cfg.agents) < 2:
        return run_mas_linear(question, metrics, cfg, api, evaluator, evaluator_lock)
    hub = cfg.agents[-1]
    spokes = cfg.agents[:-1]
    agent_ids = [agent.agent_id for agent in cfg.agents]
    memory = {agent_id: "" for agent_id in agent_ids}

    for _ in range(cfg.debate_rounds):
        for spoke in spokes:
            history = f"[{hub.agent_id}]:\n{memory[hub.agent_id]}" if memory[hub.agent_id].strip() else ""
            turn = run_agent_turn(question, spoke, history, metrics, cfg, api, evaluator, evaluator_lock)
            if turn is None:
                continue
            if turn["early_halt"]:
                return str(turn["final_ans"]), True, render_memory(memory, agent_ids)
            update_memory(memory, spoke.agent_id, turn["valid_sentences"])

        history = render_memory(memory, agent_ids)
        turn = run_agent_turn(question, hub, history, metrics, cfg, api, evaluator, evaluator_lock)
        if turn is None:
            continue
        if turn["early_halt"]:
            return str(turn["final_ans"]), True, render_memory(memory, agent_ids)
        update_memory(memory, hub.agent_id, turn["valid_sentences"])

    return "Z", False, render_memory(memory, agent_ids)


def run_mas_hierarchical(question: str, metrics: dict[str, object], cfg, api, evaluator, evaluator_lock):
    if len(cfg.agents) < 5:
        return run_mas_star(question, metrics, cfg, api, evaluator, evaluator_lock)
    leaves = cfg.agents[:3]
    mid = cfg.agents[3]
    root = cfg.agents[4]
    agent_ids = [agent.agent_id for agent in cfg.agents]
    memory = {agent_id: "" for agent_id in agent_ids}

    for _ in range(cfg.debate_rounds):
        for leaf in leaves:
            history = f"[{mid.agent_id}]:\n{memory[mid.agent_id]}" if memory[mid.agent_id].strip() else ""
            turn = run_agent_turn(question, leaf, history, metrics, cfg, api, evaluator, evaluator_lock)
            if turn is None:
                continue
            if turn["early_halt"]:
                return str(turn["final_ans"]), True, render_memory(memory, agent_ids)
            update_memory(memory, leaf.agent_id, turn["valid_sentences"])

        history = render_memory(memory, [root.agent_id] + [leaf.agent_id for leaf in leaves])
        turn = run_agent_turn(question, mid, history, metrics, cfg, api, evaluator, evaluator_lock)
        if turn is not None:
            if turn["early_halt"]:
                return str(turn["final_ans"]), True, render_memory(memory, agent_ids)
            update_memory(memory, mid.agent_id, turn["valid_sentences"])

        history = render_memory(memory, [mid.agent_id, root.agent_id])
        turn = run_agent_turn(question, root, history, metrics, cfg, api, evaluator, evaluator_lock)
        if turn is not None:
            if turn["early_halt"]:
                return str(turn["final_ans"]), True, render_memory(memory, agent_ids)
            update_memory(memory, root.agent_id, turn["valid_sentences"])

    return "Z", False, render_memory(memory, agent_ids)


def run_mas(question: str, metrics: dict[str, object], cfg, api, evaluator, evaluator_lock):
    if cfg.architecture == "linear":
        return run_mas_linear(question, metrics, cfg, api, evaluator, evaluator_lock)
    if cfg.architecture == "fullmesh":
        return run_mas_fullmesh(question, metrics, cfg, api, evaluator, evaluator_lock)
    if cfg.architecture == "star":
        return run_mas_star(question, metrics, cfg, api, evaluator, evaluator_lock)
    if cfg.architecture == "hierarchical":
        return run_mas_hierarchical(question, metrics, cfg, api, evaluator, evaluator_lock)
    raise ValueError(f"Unsupported architecture: {cfg.architecture}")


def summarize(results: list[dict[str, object]], path: str | Path, architecture: str) -> None:
    valid = [row for row in results if row["metrics"]["total_tokens"] > 0]
    if not valid:
        report = "No valid results. All API calls may have failed.\n"
    else:
        total = len(valid)
        correct = sum(1 for row in valid if row["metrics"]["is_correct"])
        total_tokens = sum(int(row["metrics"]["total_tokens"]) for row in valid)
        prompt_tokens = sum(int(row["metrics"]["prompt_tokens"]) for row in valid)
        completion_tokens = sum(int(row["metrics"]["completion_tokens"]) for row in valid)
        single = sum(1 for row in valid if row["metrics"]["route"] == "Single")
        mas = total - single
        halts = sum(1 for row in valid if row["metrics"]["early_halted"])
        halt_rate = (halts / mas * 100.0) if mas else 0.0
        report = (
            f"[{architecture}] GSM8K HALT report\n"
            f"Accuracy: {correct}/{total} ({correct / total * 100:.2f}%)\n"
            f"Routing: Single {single} | MAS {mas}\n"
            f"Early halt: {halts} ({halt_rate:.2f}% of MAS)\n"
            f"Prompt tokens: {prompt_tokens:,}\n"
            f"Completion tokens: {completion_tokens:,}\n"
            f"Total tokens: {total_tokens:,}\n"
            f"Avg tokens/question: {total_tokens / total:,.0f}\n"
        )
    print(report)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()

    from halt.evaluator import SentenceEvaluator
    from halt.router import load_router, predict_mas_probability

    api = load_api_config(args)
    cfg = load_halt_config(args.config)
    if args.architecture:
        cfg.architecture = args.architecture
    if args.router_threshold is not None:
        cfg.router_threshold = args.router_threshold

    router_embedder, router_model, trained_threshold, router_max_len = load_router(args.router)
    if args.router_threshold is None:
        cfg.router_threshold = trained_threshold
    evaluator = SentenceEvaluator(args.evaluator)
    evaluator_lock = threading.Lock()
    write_lock = threading.Lock()

    output_path = Path(args.output)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    processed = {row["global_id"] for row in read_jsonl(output_path)}
    examples = [row for row in load_gsm8k_jsonl(args.gsm8k, args.limit) if row["global_id"] not in processed]

    def process(item: dict[str, str]) -> dict[str, object] | None:
        metrics: dict[str, object] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "is_correct": False,
            "route": "Single",
            "early_halted": False,
            "mas_probability": 0.0,
        }

        mas_probability = predict_mas_probability(item["question"], router_embedder, router_model, router_max_len)
        metrics["mas_probability"] = mas_probability
        needs_mas = mas_probability >= cfg.router_threshold

        if not needs_mas:
            prompt = gsm8k_single_prompt(item["question"])
            reply, usage = call_chat_completion(prompt, cfg.final_model, api, temperature=0.0)
            add_usage(metrics, usage)
            metrics["api_calls"] = int(metrics["api_calls"]) + 1
            final_answer = extract_numeric_answer(reply)
        else:
            metrics["route"] = "MAS_ContextGC"
            final_answer, early_halted, debate_history = run_mas(
                item["question"],
                metrics,
                cfg,
                api,
                evaluator,
                evaluator_lock,
            )
            metrics["early_halted"] = early_halted
            if not early_halted:
                final_prompt = (
                    f"{item['question']}\n\n[Debate History]\n{debate_history.strip()}\n\n[Instruction]\n"
                    "Output the final numerical answer only, and put it after '#### '."
                )
                reply, usage = call_chat_completion(final_prompt, cfg.final_model, api, temperature=0.0)
                add_usage(metrics, usage)
                metrics["api_calls"] = int(metrics["api_calls"]) + 1
                final_answer = extract_numeric_answer(reply)

        metrics["is_correct"] = is_correct_number(final_answer, item["ground_truth"])
        result: dict[str, object] = {
            "global_id": item["global_id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "prediction": final_answer,
            "metrics": metrics,
        }
        with write_lock:
            append_jsonl(output_path, result)
        return result

    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process, examples), total=len(examples), desc="HALT GSM8K"):
            if result:
                results.append(result)

    all_results = read_jsonl(output_path)
    summarize(all_results, args.summary, cfg.architecture)


if __name__ == "__main__":
    main()
