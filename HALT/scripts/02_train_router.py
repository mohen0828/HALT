from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halt.config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HALT difficulty router.")
    parser.add_argument("--input", default=str(DEFAULT_DATA_DIR / "router_training_data.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT_DIR / "router_gsm8k.pkl"))
    parser.add_argument("--embedder", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from halt.router import build_router_features, save_router

    records = read_jsonl(args.input)
    if not records:
        raise ValueError(f"No router records found in {args.input}")

    questions = [str(row["question"]) for row in records]
    labels = np.array([1 if row.get("needs_mas") else 0 for row in records])
    if len(set(labels.tolist())) < 2:
        raise ValueError("Router data must contain both classes: needs_mas=true and false.")

    embedder = SentenceTransformer(args.embedder)
    features, max_length = build_router_features(questions, embedder)

    stratify = labels if min(np.bincount(labels)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        class_weight={0: 1.0, 1: 3.0},
        random_state=42,
    )
    clf.fit(x_train, y_train)

    y_prob = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= args.threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=["single", "mas"]))
    print("confusion_matrix:")
    print(confusion_matrix(y_test, y_pred))

    save_router(Path(args.output), clf, args.embedder, args.threshold, max_length)
    print(f"Saved router to {args.output}")


if __name__ == "__main__":
    main()
