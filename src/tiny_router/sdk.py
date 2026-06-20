from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, TypeVar

from .config import ModelTarget, RouterConfig
from .errors import ExhaustedError, InvalidPromptError, ProviderError
from .model import RouterModel
from .types import RouteDecision, Tier


@dataclass(frozen=True)
class RoutingResult:
    decision: RouteDecision
    target: ModelTarget

    @property
    def tier(self) -> Tier:
        return self.decision.tier

    @property
    def model(self) -> str:
        return self.target.model

    def to_dict(self) -> dict[str, Any]:
        result = self.decision.to_dict()
        result["model"] = self.target.model
        if self.target.provider is not None:
            result["provider"] = self.target.provider
        if self.target.metadata:
            result["model_metadata"] = dict(self.target.metadata)
        return result


T = TypeVar("T")


@dataclass(frozen=True)
class ExecutionAttempt:
    tier: Tier
    target: ModelTarget
    accepted: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.label,
            "model": self.target.model,
            "accepted": self.accepted,
            "error": self.error,
        }


@dataclass(frozen=True)
class ExecutionResult(Generic[T]):
    output: T
    route: RoutingResult
    attempts: tuple[ExecutionAttempt, ...]

    @property
    def escalated(self) -> bool:
        return len(self.attempts) > 1


class Router:
    """Thread-safe, provider-agnostic routing SDK."""

    def __init__(self, model: RouterModel, config: RouterConfig, *, max_prompt_chars: int = 200_000) -> None:
        if not isinstance(max_prompt_chars, int) or isinstance(max_prompt_chars, bool) or max_prompt_chars < 1:
            raise ValueError("max_prompt_chars must be a positive integer")
        self.model = model
        self.config = config
        self.max_prompt_chars = max_prompt_chars

    @classmethod
    def from_files(
        cls, model_path: str | Path, config_path: str | Path, *, max_prompt_chars: int = 200_000
    ) -> "Router":
        return cls(RouterModel.load(model_path), RouterConfig.load(config_path), max_prompt_chars=max_prompt_chars)

    def route(
        self,
        prompt: str,
        *,
        minimum_tier: Tier | str | int | None = None,
        maximum_tier: Tier | str | int | None = None,
    ) -> RoutingResult:
        self._validate_prompt(prompt)
        policy = self.config.policy
        overrides: dict[str, Tier] = {}
        if minimum_tier is not None:
            overrides["minimum_tier"] = Tier.parse(minimum_tier)
        if maximum_tier is not None:
            overrides["maximum_tier"] = Tier.parse(maximum_tier)
        if overrides:
            try:
                policy = replace(policy, **overrides)
            except ValueError as exc:
                raise InvalidPromptError(f"invalid request tier bounds: {exc}") from exc
        decision = policy.route(self.model, prompt)
        return RoutingResult(decision, self.config.target_for(decision.tier))

    def route_many(self, prompts: Iterable[str]) -> list[RoutingResult]:
        if isinstance(prompts, (str, bytes)):
            raise TypeError("prompts must be an iterable of strings, not a single string")
        return [self.route(prompt) for prompt in prompts]

    def execute(
        self,
        prompt: str,
        invoke: Callable[[ModelTarget, str], T],
        *,
        validate: Callable[[T], bool] | None = None,
        maximum_tier: Tier | str | int | None = None,
    ) -> ExecutionResult[T]:
        """Route, invoke, validate, and escalate until an answer is accepted."""
        initial = self.route(prompt, maximum_tier=maximum_tier)
        ceiling = Tier.parse(maximum_tier) if maximum_tier is not None else self.config.policy.maximum_tier
        attempts: list[ExecutionAttempt] = []
        for tier_value in range(int(initial.tier), int(ceiling) + 1):
            tier = Tier(tier_value)
            target = self.config.target_for(tier)
            try:
                output = invoke(target, prompt)
            except ProviderError as exc:
                attempts.append(ExecutionAttempt(tier, target, False, str(exc)))
                if not exc.retryable:
                    raise
                continue
            accepted = validate(output) if validate is not None else True
            attempts.append(ExecutionAttempt(tier, target, accepted))
            if accepted:
                decision = replace(initial.decision, tier=tier, reason=(
                    initial.decision.reason if tier == initial.tier else "response_validation_escalation"
                ))
                return ExecutionResult(output, RoutingResult(decision, target), tuple(attempts))
        raise ExhaustedError("no model tier produced an acceptable response", attempts=tuple(attempts))

    def _validate_prompt(self, prompt: str) -> None:
        if not isinstance(prompt, str):
            raise InvalidPromptError("prompt must be a string")
        if not prompt.strip():
            raise InvalidPromptError("prompt must not be empty")
        if len(prompt) > self.max_prompt_chars:
            raise InvalidPromptError(f"prompt exceeds max_prompt_chars={self.max_prompt_chars}")
