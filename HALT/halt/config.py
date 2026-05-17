from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "gsm8k"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class ApiConfig:
    api_url: str
    api_key: str
    timeout: int = 90
    max_retries: int = 3


@dataclass
class AgentConfig:
    agent_id: str
    model: str
    role_prompt: str


@dataclass
class HaltConfig:
    single_model: str = "gpt-4o-mini"
    final_model: str = "gpt-4o-mini"
    oracle_model: str = "gpt-4o-mini"
    agents: list[AgentConfig] = field(default_factory=list)
    debate_rounds: int = 2
    router_threshold: float = 0.5
    early_halt_max_threshold: float = 0.85
    early_halt_min_threshold: float = 0.45
    prune_keep_threshold: float = 0.6
    architecture: str = "linear"


def default_agents(model: str = "gpt-4o-mini") -> list[AgentConfig]:
    return [
        AgentConfig(
            agent_id="Agent_A_Explorer",
            model=model,
            role_prompt=(
                "You are the Explorer. Based on the problem and current discussion, "
                "propose clear step-by-step reasoning and identify the core quantities."
            ),
        ),
        AgentConfig(
            agent_id="Agent_B_Critic",
            model=model,
            role_prompt=(
                "You are the Critic. Strictly scrutinize previous reasoning, find "
                "logical gaps or calculation errors, and correct them directly."
            ),
        ),
        AgentConfig(
            agent_id="Agent_C_Lateral_Thinker",
            model=model,
            role_prompt=(
                "You are the Lateral Thinker. Solve or verify the problem from an "
                "alternative angle to cross-check previous steps."
            ),
        ),
        AgentConfig(
            agent_id="Agent_D_Verifier",
            model=model,
            role_prompt=(
                "You are the Verifier. Recalculate the math carefully and check that "
                "the final numerical answer follows from the reasoning."
            ),
        ),
        AgentConfig(
            agent_id="Agent_E_Synthesizer",
            model=model,
            role_prompt=(
                "You are the Synthesizer. Resolve disagreements, keep only reliable "
                "reasoning, and push toward a concise final solution."
            ),
        ),
    ]


def load_halt_config(path: str | Path | None) -> HaltConfig:
    cfg = HaltConfig(agents=default_agents())
    if not path:
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for key in [
        "single_model",
        "final_model",
        "oracle_model",
        "debate_rounds",
        "router_threshold",
        "early_halt_max_threshold",
        "early_halt_min_threshold",
        "prune_keep_threshold",
        "architecture",
    ]:
        if key in raw:
            setattr(cfg, key, raw[key])

    if "agents" in raw:
        cfg.agents = [AgentConfig(**item) for item in raw["agents"]]

    return cfg


def load_api_config(args: argparse.Namespace) -> ApiConfig:
    api_url = args.api_url or os.getenv("OPENAI_BASE_URL") or os.getenv("HALT_API_URL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("HALT_API_KEY")
    if not api_url:
        raise ValueError("Missing API URL. Pass --api-url or set OPENAI_BASE_URL/HALT_API_URL.")
    if not api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY/HALT_API_KEY.")
    return ApiConfig(
        api_url=api_url,
        api_key=api_key,
        timeout=getattr(args, "timeout", 90),
        max_retries=getattr(args, "max_retries", 3),
    )


def add_common_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer environment variables for release use.")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=3)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return records
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

