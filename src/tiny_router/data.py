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
    group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise DatasetError("example prompt must be a non-empty string")
        if not isinstance(self.label, Tier):
            raise DatasetError("example label must be a Tier")
        if (
            not isinstance(self.weight, (int, float))
            or isinstance(self.weight, bool)
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise DatasetError("example weight must be finite and positive")
        if self.group is not None and (not isinstance(self.group, str) or not self.group.strip()):
            raise DatasetError("example group must be a non-empty string when set")


def label_from_scores(scores: dict[str, float], acceptable_score: float) -> Tier:
    if (
        isinstance(acceptable_score, bool)
        or not isinstance(acceptable_score, (int, float))
        or not math.isfinite(acceptable_score)
        or not 0 <= acceptable_score <= 1
    ):
        raise DatasetError("acceptable_score must be finite and in [0, 1]")
    validated: dict[Tier, float] = {}
    for tier in Tier:
        if tier.label not in scores:
            raise DatasetError(f"scores.{tier.label} is required")
        raw_score = scores[tier.label]
        try:
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise TypeError
            score = float(raw_score)
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
    raw_weight = record.get("weight", 1.0)
    try:
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise DatasetError("record.weight must be numeric") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise DatasetError("record.weight must be finite and positive")
    group = record.get("group")
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise DatasetError("record.group must be a non-empty string when set")
    return Example(prompt=prompt, label=label, weight=weight, group=group)


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
    items = list(examples)
    if any(example.group is not None for example in items):
        return _split_grouped(items, validation_fraction, seed)
    buckets = {tier: [] for tier in Tier}
    for example in items:
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


def _split_grouped(
    items: list[Example], validation_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    groups: dict[str, list[Example]] = {}
    for index, example in enumerate(items):
        key = example.group if example.group is not None else f"\x00ungrouped:{index}"
        groups.setdefault(key, []).append(example)

    rng = random.Random(seed)
    candidates = list(groups.items())
    rng.shuffle(candidates)
    tier_totals = {tier: sum(example.label == tier for example in items) for tier in Tier}
    tier_targets = {tier: tier_totals[tier] * validation_fraction for tier in Tier}
    target_size = int(len(items) * validation_fraction)
    if validation_fraction and len(items) > 1:
        target_size = max(1, target_size)
    selected: set[str] = set()
    counts = {tier: 0 for tier in Tier}

    def group_counts(group: list[Example]) -> dict[Tier, int]:
        return {tier: sum(example.label == tier for example in group) for tier in Tier}

    while candidates and sum(counts.values()) < target_size:
        best_index: int | None = None
        best_key: tuple[float, float] | None = None
        for index, (_, group) in enumerate(candidates):
            additions = group_counts(group)
            if any(
                tier_totals[tier] > 1 and counts[tier] + additions[tier] >= tier_totals[tier]
                for tier in Tier
            ):
                continue
            before = sum(abs(counts[tier] - tier_targets[tier]) for tier in Tier)
            after = sum(abs(counts[tier] + additions[tier] - tier_targets[tier]) for tier in Tier)
            new_total = sum(counts.values()) + len(group)
            key = (before - after, -abs(new_total - target_size))
            if best_key is None or key > best_key:
                best_key, best_index = key, index
        if best_index is None:
            break
        name, group = candidates.pop(best_index)
        selected.add(name)
        additions = group_counts(group)
        for tier in Tier:
            counts[tier] += additions[tier]

    validation = [example for index, example in enumerate(items) if (example.group or f"\x00ungrouped:{index}") in selected]
    training = [example for index, example in enumerate(items) if (example.group or f"\x00ungrouped:{index}") not in selected]
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation
