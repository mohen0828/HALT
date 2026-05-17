from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .text import score_input_text


class SentenceEvaluator:
    def __init__(self, model_dir: str | Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def score(self, question: str, agent_id: str, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        texts = [score_input_text(question, agent_id, sentence) for sentence in sentences]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=256, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return [max(0.0, min(1.0, float(score))) for score in scores]

