# HALT GSM8K Example

This folder is a cleaned open-source version of the HALT workflow using GSM8K as the running example.
It does not include trained router/evaluator weights or generated datasets.

HALT combines:

1. a difficulty router that decides whether a question should use a single model or MAS,
2. a multi-agent solver,
3. a sentence evaluator that prunes weak reasoning and triggers early halt,
4. an end-to-end runner that records accuracy, routing, early halt rate, and token usage.

## Layout

```text
HALT/
  configs/gsm8k.example.json        # model roster and HALT thresholds
  halt/                             # shared package code
  scripts/
    01_collect_router_data.py       # single-model runs -> router labels
    02_train_router.py              # train sklearn router
    03_collect_mas_traces.py        # full MAS traces for evaluator data
    04_oracle_score_traces.py       # oracle LLM labels each sentence
    05_prepare_evaluator_data.py    # scored traces -> CSV
    06_train_evaluator.py           # train DeBERTa regression evaluator
    07_run_halt_gsm8k.py            # end-to-end HALT evaluation
  data/gsm8k/                       # generated jsonl/csv files, ignored by git
  artifacts/                        # trained models, ignored by git
  outputs/                          # evaluation outputs, ignored by git
```

## Setup

Install dependencies from this directory:

```powershell
pip install -r requirements.txt
```

Set an OpenAI-compatible endpoint and key:

```powershell
$env:OPENAI_BASE_URL="https://api.openai.com/v1/chat/completions"
$env:OPENAI_API_KEY="..."
```

Use any OpenAI-compatible provider by changing `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and the model names in
`configs/gsm8k.example.json`.

## Data

The scripts expect a GSM8K JSONL file with:

```json
{"question": "...", "answer": "... #### 42"}
```

Pass its path with `--gsm8k`. No dataset file is copied into this repository.

## Workflow

Run the commands from the `HALT` directory.

Collect router labels:

```powershell
python scripts/01_collect_router_data.py --config configs/gsm8k.example.json --gsm8k path\to\gsm8k.jsonl --limit 500
```

Train the router:

```powershell
python scripts/02_train_router.py
```

Collect full MAS traces for examples that the single model missed:

```powershell
python scripts/03_collect_mas_traces.py --config configs/gsm8k.example.json --workers 2
```

Score each MAS sentence with an oracle model:

```powershell
python scripts/04_oracle_score_traces.py --config configs/gsm8k.example.json --workers 4
```

Prepare evaluator training data:

```powershell
python scripts/05_prepare_evaluator_data.py --balance
```

Train the evaluator:

```powershell
python scripts/06_train_evaluator.py
```

Run HALT end to end:

```powershell
python scripts/07_run_halt_gsm8k.py --config configs/gsm8k.example.json --gsm8k path\to\gsm8k.jsonl --architecture linear --limit 200 --overwrite
```

Supported architectures are `linear`, `fullmesh`, `star`, and `hierarchical`.

## Notes For Release

- API keys are read from environment variables or CLI flags. Do not commit secrets.
- `artifacts/`, `data/gsm8k/`, and `outputs/` are intentionally ignored.
- The default router uses `BAAI/bge-large-en-v1.5`.
- The evaluator is a regression model over scores in `[0, 1]`, defaulting to `microsoft/deberta-v3-small`.
- If you already have trained parameters, place them under `artifacts/` locally, but do not add them to the open-source package unless the release explicitly includes model weights.
