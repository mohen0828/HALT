from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halt.config import DEFAULT_DATA_DIR, read_jsonl
from halt.text import score_input_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert scored traces into evaluator training CSV.")
    parser.add_argument("--input", default=str(DEFAULT_DATA_DIR / "scored_traces.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR / "evaluator_training.csv"))
    parser.add_argument("--balanced-output", default=str(DEFAULT_DATA_DIR / "evaluator_training_balanced.csv"))
    parser.add_argument("--balance", action="store_true")
    return parser.parse_args()


def score_lookup(scores: list[dict[str, object]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for row in scores:
        sid = row.get("sentence_id") or row.get("id") or row.get("sentenceId")
        if not sid:
            continue
        try:
            lookup[str(sid)] = float(row.get("score", 0.5))
        except (TypeError, ValueError):
            lookup[str(sid)] = 0.5
    return lookup


def prepare_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in records:
        scores = score_lookup(item.get("fine_grained_scores", []))
        mapping = item.get("sentence_mapping", {})
        turn_to_agent = {int(t["turn"]): str(t["agent"]) for t in item.get("mas_transcript", [])}
        for sid, sentence in mapping.items():
            try:
                turn_idx = int(str(sid).split("_")[0].replace("T", ""))
            except ValueError:
                continue
            agent = turn_to_agent.get(turn_idx, "Agent")
            rows.append(
                {
                    "global_id": item["global_id"],
                    "sentence_id": sid,
                    "input_text": score_input_text(str(item["question"]), agent, str(sentence)),
                    "label_score": scores.get(str(sid), 0.5),
                }
            )
    return rows


def balance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Keep all minority labels and downsample frequent neutral/helpful labels.
    fractions = {0.0: 0.6, 0.25: 1.0, 0.5: 0.1, 0.75: 0.2, 1.0: 0.4}
    parts = []
    for label, group in df.groupby("label_score"):
        frac = fractions.get(float(label), 1.0)
        parts.append(group.sample(frac=min(1.0, frac), random_state=42))
    return pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    rows = prepare_rows(records)
    if not rows:
        raise ValueError(f"No evaluator examples could be extracted from {args.input}")

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(df)} evaluator rows to {args.output}")
    print(df["label_score"].value_counts(normalize=True).sort_index())

    if args.balance:
        balanced = balance_dataframe(df)
        Path(args.balanced_output).parent.mkdir(parents=True, exist_ok=True)
        balanced.to_csv(args.balanced_output, index=False, encoding="utf-8")
        print(f"Saved {len(balanced)} balanced rows to {args.balanced_output}")
        print(balanced["label_score"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
