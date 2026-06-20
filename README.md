# Tiny Capability Router

A dependency-free Python classifier and SDK that routes a prompt to the cheapest model tier likely to answer it correctly.

```text
prompt -> calibrated classifier -> cost/risk policy -> model target
                                             |
                                             +-> validate response -> escalate
```

The classifier estimates the **minimum capable tier** (`low`, `medium`, `high`). The policy separately minimizes model cost plus expected under-routing damage. Keeping those concerns separate lets you change prices and risk tolerance without retraining.

## Status

This repository contains a complete training, routing, benchmarking, SDK, and HTTP pipeline. The included seed dataset is a smoke-test fixture, not a production benchmark. Real quality depends on representative prompts scored against your actual models.

Runtime dependencies: none. Python 3.10+.

## Quick start

```bash
make test
make train
PYTHONPATH=src python3 -m tiny_router init
PYTHONPATH=src python3 -m tiny_router route \
  --model model.json --config router.json \
  "Find the race condition and justify the fix"
```

Or install the CLI and SDK:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
tiny-router --version
```

## Python SDK

Configure which provider/model serves each tier:

```json
{
  "models": {
    "low": {"provider": "your-client", "model": "small-model"},
    "medium": {"provider": "your-client", "model": "standard-model"},
    "high": {"provider": "your-client", "model": "frontier-model"}
  },
  "policy": {
    "tier_costs": [1, 5, 20],
    "underroute_penalty": 30,
    "confidence_threshold": 0.45,
    "uncertain_tier": "medium",
    "minimum_tier": "low",
    "maximum_tier": "high",
    "high_probability_threshold": 0.7
  }
}
```

Route without invoking a provider:

```python
from tiny_router import Router

router = Router.from_files("model.json", "router.json")
result = router.route("Prove this concurrent queue is linearizable")

print(result.tier.label, result.model, result.decision.confidence)
```

Per-request safety bounds are explicit:

```python
result = router.route(prompt, minimum_tier="medium", maximum_tier="high")
```

### Invoke and escalate

The SDK stays provider-agnostic. Pass your own model client and optional response validator:

```python
from tiny_router import ModelTarget, ProviderError

def invoke(target: ModelTarget, prompt: str) -> str:
    try:
        return my_client.generate(model=target.model, prompt=prompt)
    except TimeoutError as exc:
        raise ProviderError(str(exc), retryable=True) from exc

result = router.execute(
    prompt,
    invoke,
    validate=lambda answer: answer_passes_task_checks(answer),
)

print(result.output, result.route.model, result.escalated)
```

Rejected answers and retryable provider failures move up one tier. Non-retryable failures remain visible. `execute_async` provides the same behavior for async clients and accepts sync or async validators.

## Build a real dataset

Do not label difficulty by intuition. Run every representative prompt through all three model tiers, judge each answer, and choose the cheapest tier meeting the acceptance threshold.

```python
from tiny_router import BenchmarkTask, RouterConfig, run_benchmark, write_benchmark_jsonl

config = RouterConfig.load("router.json")
tasks = [BenchmarkTask(prompt, reference=expected, group=task_family)]

records = run_benchmark(
    tasks,
    config,
    invoke=lambda target, prompt: client.generate(target.model, prompt),
    judge=lambda task, answer: score_answer(task.reference, answer),  # 0..1
)
write_benchmark_jsonl(records, "data/benchmark.jsonl")
```

`run_benchmark_async` supports async providers and judges. Provider failures are recorded as score `0` with latency and error text. The resulting JSONL feeds directly into training:

```bash
tiny-router train data/benchmark.jsonl --output model.json --acceptable-score 0.8
tiny-router evaluate data/held-out.jsonl --model model.json
```

Use `group` for paraphrases or variants of the same underlying task. Grouped examples are kept on one side of the train/validation split to prevent leakage.

Direct labels are also accepted:

```json
{"prompt":"Translate hello to French","label":"low","group":"translation-1","weight":1.0}
```

Score records require all tiers. If no tier reaches the acceptable score, loading fails instead of falsely labeling the prompt `high`.

## Evaluation

Reports include:

- classifier accuracy and policy exact-tier accuracy;
- under-route and over-route rates;
- mean capability shortfall;
- selected cost, realized cost, and regret against the cheapest sufficient tier;
- savings versus always selecting high;
- log loss, Brier score, per-tier recall, and confusion matrix.

Training fits temperature on held-out examples so probabilities are useful to the cost policy. For production, keep a separate locked test set; the automatic validation split is for iteration, not final claims.

## HTTP API

```bash
tiny-router serve --model model.json --config router.json --port 8080
```

Endpoints:

- `GET /health`
- `GET /models`
- `POST /route` with `{"prompt":"...","minimum_tier":"low","maximum_tier":"high"}`
- `POST /route/batch` with `{"prompts":["...","..."]}` (maximum 1,000)

The server accepts JSON bodies up to 1 MB, validates prompts up to 200,000 characters, and returns stable JSON error objects. It only selects models; provider invocation stays in your application where credentials, retries, budgets, and observability belong.

## CLI

```bash
tiny-router init [--output router.json] [--force]
tiny-router train DATASET [--output model.json] [--no-calibrate]
tiny-router evaluate DATASET --model model.json
tiny-router route --model model.json [--config router.json] [PROMPT]
tiny-router inspect --model model.json
tiny-router serve --model model.json [--config router.json]
```

Omit `PROMPT` to read stdin. Model artifacts are checksum-protected, bounded to 64 MB when read, and written atomically.

## Fuzzing and tests

```bash
make test
python3 -m pip install -e '.[test]'
python3 -m pytest
```

The standard-library suite fuzzes arbitrary Unicode/control-character prompts, thousands of probability distributions, malformed model/config JSON, and policy type boundaries. Installing the test extra adds 500 Hypothesis-generated Unicode cases.

## Production checklist

1. Collect prompts from the workload you will actually route.
2. Use task-specific tests or carefully validated human/judge scoring.
3. Group related prompt variants before splitting.
4. Lock an untouched test set and define a maximum under-route rate.
5. Tune policy costs from measured price, latency, and failure impact.
6. Log selected tier, confidence, escalation, outcome, latency, and cost.
7. Re-benchmark when any underlying model changes.
