from __future__ import annotations

import math
from dataclasses import dataclass

from .model import RouterModel
from .types import RouteDecision, Tier


@dataclass(frozen=True)
class RoutingPolicy:
    tier_costs: tuple[float, float, float] = (1.0, 4.0, 15.0)
    underroute_penalty: float = 25.0
    confidence_threshold: float = 0.45
    uncertain_tier: Tier = Tier.MEDIUM
    minimum_tier: Tier = Tier.LOW
    maximum_tier: Tier = Tier.HIGH
    high_probability_threshold: float = 0.70

    def __post_init__(self) -> None:
        if len(self.tier_costs) != 3 or any(cost < 0 or not math.isfinite(cost) for cost in self.tier_costs):
            raise ValueError("tier costs must be three finite non-negative numbers")
        if tuple(sorted(self.tier_costs)) != self.tier_costs:
            raise ValueError("tier costs must be non-decreasing")
        if self.underroute_penalty < 0 or not math.isfinite(self.underroute_penalty):
            raise ValueError("underroute penalty must be finite and non-negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence threshold must be in [0, 1]")
        if self.minimum_tier > self.maximum_tier:
            raise ValueError("minimum_tier cannot exceed maximum_tier")
        if not 0 <= self.high_probability_threshold <= 1:
            raise ValueError("high_probability_threshold must be in [0, 1]")

    def expected_costs(self, probabilities: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(probabilities) != 3 or any(value < 0 or not math.isfinite(value) for value in probabilities):
            raise ValueError("probabilities must be three finite non-negative numbers")
        total = sum(probabilities)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        costs = []
        for selected in Tier:
            expected_failure = sum(
                probability * max(0, int(required) - int(selected))
                for required, probability in zip(Tier, probabilities)
            )
            costs.append(self.tier_costs[int(selected)] + self.underroute_penalty * expected_failure)
        return tuple(costs)  # type: ignore[return-value]

    def route(self, model: RouterModel, prompt: str) -> RouteDecision:
        probabilities = model.predict_proba(prompt)
        return self.route_probabilities(probabilities)

    def route_probabilities(self, probabilities: tuple[float, float, float]) -> RouteDecision:
        confidence = max(probabilities)
        expected = self.expected_costs(probabilities)
        candidates = range(int(self.minimum_tier), int(self.maximum_tier) + 1)
        tier = Tier(min(candidates, key=expected.__getitem__))
        reason = "minimum_expected_cost"
        if probabilities[int(Tier.HIGH)] >= self.high_probability_threshold and self.maximum_tier >= Tier.HIGH:
            tier = Tier.HIGH
            reason = "high_risk_guardrail"
        fallback_tier = min(self.maximum_tier, max(self.minimum_tier, self.uncertain_tier))
        if confidence < self.confidence_threshold and tier < fallback_tier:
            tier = fallback_tier
            reason = "low_confidence_fallback"
        return RouteDecision(tier, probabilities, confidence, expected, reason)
