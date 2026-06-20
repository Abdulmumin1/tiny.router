from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ConfigurationError
from .policy import RoutingPolicy
from .types import Tier


@dataclass(frozen=True)
class ModelTarget:
    model: str
    provider: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigurationError("model identifier must be a non-empty string")
        if self.provider is not None and (not isinstance(self.provider, str) or not self.provider.strip()):
            raise ConfigurationError("provider must be a non-empty string when set")
        if not isinstance(self.metadata, Mapping):
            raise ConfigurationError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def parse(cls, value: object) -> "ModelTarget":
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, dict):
            raise ConfigurationError("model target must be a string or object")
        unknown = set(value) - {"model", "provider", "metadata"}
        if unknown:
            raise ConfigurationError(f"unknown model target keys: {sorted(map(str, unknown))}")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ConfigurationError("model target metadata must be an object")
        return cls(model=value.get("model"), provider=value.get("provider"), metadata=metadata)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"model": self.model}
        if self.provider is not None:
            result["provider"] = self.provider
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class RouterConfig:
    models: Mapping[Tier, ModelTarget]
    policy: RoutingPolicy = field(default_factory=RoutingPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.models, Mapping):
            raise ConfigurationError("models must be a mapping")
        missing = set(Tier) - set(self.models)
        extra = set(self.models) - set(Tier)
        if missing or extra:
            raise ConfigurationError(
                f"models must define exactly low, medium, high; missing={sorted(t.label for t in missing)}"
            )
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))

    def target_for(self, tier: Tier | str | int) -> ModelTarget:
        return self.models[Tier.parse(tier)]

    @classmethod
    def from_dict(cls, payload: object) -> "RouterConfig":
        if not isinstance(payload, dict):
            raise ConfigurationError("configuration must be an object")
        unknown = set(payload) - {"models", "policy"}
        if unknown:
            raise ConfigurationError(f"unknown configuration keys: {sorted(map(str, unknown))}")
        raw_models = payload.get("models")
        if not isinstance(raw_models, dict):
            raise ConfigurationError("models must be an object")
        expected_model_keys = {tier.label for tier in Tier}
        if set(raw_models) != expected_model_keys:
            raise ConfigurationError("model keys must be exactly: low, medium, high")
        try:
            models = {Tier.parse(key): ModelTarget.parse(value) for key, value in raw_models.items()}
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(str(exc)) from exc

        raw_policy = payload.get("policy", {})
        if not isinstance(raw_policy, dict):
            raise ConfigurationError("policy must be an object")
        policy_fields = {item.name for item in fields(RoutingPolicy)}
        policy_unknown = set(raw_policy) - policy_fields
        if policy_unknown:
            raise ConfigurationError(f"unknown policy keys: {sorted(map(str, policy_unknown))}")
        converted = dict(raw_policy)
        for name in ("uncertain_tier", "minimum_tier", "maximum_tier"):
            if name in converted:
                converted[name] = Tier.parse(converted[name])
        if "tier_costs" in converted:
            converted["tier_costs"] = tuple(converted["tier_costs"])
        try:
            policy = RoutingPolicy(**converted)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid policy: {exc}") from exc
        return cls(models=models, policy=policy)

    @classmethod
    def load(cls, path: str | Path) -> "RouterConfig":
        source = Path(path)
        try:
            return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        except ConfigurationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {source}: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "models": {tier.label: self.models[tier].to_dict() for tier in Tier},
            "policy": {
                "tier_costs": list(self.policy.tier_costs),
                "underroute_penalty": self.policy.underroute_penalty,
                "confidence_threshold": self.policy.confidence_threshold,
                "uncertain_tier": self.policy.uncertain_tier.label,
                "minimum_tier": self.policy.minimum_tier.label,
                "maximum_tier": self.policy.maximum_tier.label,
                "high_probability_threshold": self.policy.high_probability_threshold,
            },
        }
