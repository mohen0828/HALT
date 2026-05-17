from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def build_router_features(
    questions: list[str],
    embedder: SentenceTransformer,
    max_length: float | None = None,
) -> tuple[np.ndarray, float]:
    embeddings = embedder.encode(questions, show_progress_bar=True)
    if max_length is None:
        max_length = float(max(len(q) for q in questions) or 1)
    lengths = np.array([[len(q) / max_length] for q in questions], dtype=np.float32)
    return np.hstack((embeddings, lengths)), max_length


def save_router(path: str | Path, model: object, embedder_name: str, threshold: float, max_length: float) -> None:
    payload = {
        "model": model,
        "embedder_name": embedder_name,
        "threshold": threshold,
        "max_length": max_length,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_router(path: str | Path) -> tuple[SentenceTransformer, object, float, float]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    embedder = SentenceTransformer(payload["embedder_name"])
    return (
        embedder,
        payload["model"],
        float(payload.get("threshold", 0.5)),
        float(payload.get("max_length", 2000.0)),
    )


def predict_mas_probability(question: str, embedder: SentenceTransformer, model: object, max_length: float) -> float:
    emb = embedder.encode([question], show_progress_bar=False)
    length_feat = np.array([[len(question) / max_length]], dtype=np.float32)
    features = np.hstack((emb, length_feat))
    return float(model.predict_proba(features)[0][1])

