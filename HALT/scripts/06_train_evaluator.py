from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halt.config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HALT sentence evaluator.")
    parser.add_argument("--input", default=str(DEFAULT_DATA_DIR / "evaluator_training_balanced.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACT_DIR / "evaluator_deberta_v3"))
    parser.add_argument("--base-model", default="microsoft/deberta-v3-small")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    return parser.parse_args()


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    predictions, labels = eval_pred
    predictions = np.clip(predictions.squeeze(), 0.0, 1.0)
    toxic_labels = labels <= 0.25
    toxic_preds = predictions <= 0.35
    toxic_recall = float(np.sum(toxic_labels & toxic_preds) / np.sum(toxic_labels)) if np.any(toxic_labels) else 0.0
    return {
        "mse": mean_squared_error(labels, predictions),
        "mae": mean_absolute_error(labels, predictions),
        "toxic_recall": toxic_recall,
    }


def main() -> None:
    args = parse_args()

    import pandas as pd
    import torch
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    df = pd.read_csv(args.input)
    if args.sample_fraction < 1.0:
        df = df.sample(frac=args.sample_fraction, random_state=42)
    df = df.rename(columns={"label_score": "labels"})
    df["labels"] = df["labels"].astype(float)

    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=256)

    remove_cols = [col for col in train_ds.column_names if col != "labels"]
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=remove_cols)
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=remove_cols)

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved evaluator to {args.output_dir}")


if __name__ == "__main__":
    main()
