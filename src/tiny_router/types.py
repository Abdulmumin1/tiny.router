from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Tier(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @classmethod
    def parse(cls, value: str | int | "Tier") -> "Tier":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError(f"unknown tier: {value!r}")
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(f"unknown tier: {value!r}") from exc
        try:
            return cls[value.strip().upper()]
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"unknown tier: {value!r}") from exc

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class RouteDecision:
    tier: Tier
    probabilities: tuple[float, float, float]
    confidence: float
    expected_costs: tuple[float, float, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.label,
            "confidence": self.confidence,
            "probabilities": {
                tier.label: self.probabilities[int(tier)] for tier in Tier
            },
            "expected_costs": {
                tier.label: self.expected_costs[int(tier)] for tier in Tier
            },
            "reason": self.reason,
        }
