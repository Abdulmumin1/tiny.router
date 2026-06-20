from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import Tier
from .errors import DatasetError


@dataclass(frozen=True)
class Example:
    prompt: str
    label: Tier
    weight: float = 1.0


def label_from_scores(scores: dict[str, float], acceptable_score: float) -> Tier:
    if not math.isfinite(acceptable_score) or not 0 <= acceptable_score <= 1:
        raise DatasetError("acceptable_score must be finite and in [0, 1]")
    validated: dict[Tier, float] = {}
    for tier in Tier:
        if tier.label not in scores:
            raise DatasetError(f"scores.{tier.label} is required")
        try:
            score = float(scores[tier.label])
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"scores.{tier.label} must be numeric") from exc
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise DatasetError(f"scores.{tier.label} must be finite and in [0, 1]")
        validated[tier] = score
    for tier in Tier:
        if validated[tier] >= acceptable_score:
            return tier
    raise DatasetError("no tier meets the acceptable score")


def parse_record(record: dict[str, object], acceptable_score: float = 0.8) -> Example:
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("record.prompt must be a non-empty string")
    if "label" in record:
        label = Tier.parse(record["label"])  # type: ignore[arg-type]
    else:
        scores = record.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("record needs either label or scores")
        label = label_from_scores(scores, acceptable_score)  # type: ignore[arg-type]
    try:
        weight = float(record.get("weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise DatasetError("record.weight must be numeric") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise DatasetError("record.weight must be finite and positive")
    return Example(prompt=prompt, label=label, weight=weight)


def load_jsonl(path: str | Path, acceptable_score: float = 0.8) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                examples.append(parse_record(record, acceptable_score))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise DatasetError(f"{path}:{line_number}: {exc}") from exc
    if not examples:
        raise DatasetError(f"{path}: dataset is empty")
    return examples


def split_examples(
    examples: Iterable[Example], validation_fraction: float = 0.2, seed: int = 17
) -> tuple[list[Example], list[Example]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    buckets = {tier: [] for tier in Tier}
    for example in examples:
        buckets[example.label].append(example)
    rng = random.Random(seed)
    total_items = sum(len(bucket) for bucket in buckets.values())
    target_size = int(total_items * validation_fraction)
    if validation_fraction and total_items > 1:
        target_size = max(1, target_size)
    allocations = {
        tier: min(int(len(bucket) * validation_fraction), max(0, len(bucket) - 1))
        for tier, bucket in buckets.items()
    }
    while sum(allocations.values()) < target_size:
        candidates = [tier for tier in Tier if allocations[tier] < max(0, len(buckets[tier]) - 1)]
        if not candidates:
            break
        tier = max(candidates, key=lambda candidate: len(buckets[candidate]) * validation_fraction - allocations[candidate])
        allocations[tier] += 1

    training: list[Example] = []
    validation: list[Example] = []
    for tier in Tier:
        bucket = buckets[tier]
        rng.shuffle(bucket)
        size = allocations[tier]
        validation.extend(bucket[:size])
        training.extend(bucket[size:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation
