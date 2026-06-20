from __future__ import annotations

import json
import hashlib
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .data import Example
from .errors import ArtifactError
from .features import extract_features
from .types import Tier

MODEL_FORMAT = "tiny-router-v2"
LEGACY_FORMAT = "tiny-router-v1"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def _softmax(logits: Sequence[float]) -> tuple[float, float, float]:
    maximum = max(logits)
    exponents = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in logits]
    total = sum(exponents)
    return tuple(value / total for value in exponents)  # type: ignore[return-value]


@dataclass
class RouterModel:
    dimensions: int
    weights: list[list[float]]
    temperature: float = 1.0

    @classmethod
    def empty(cls, dimensions: int = 1024) -> "RouterModel":
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        return cls(dimensions, [[0.0] * dimensions for _ in Tier])

    def predict_proba(self, prompt: str) -> tuple[float, float, float]:
        features = extract_features(prompt, self.dimensions)
        temperature = max(self.temperature, 1e-3)
        logits = [
            sum(row[index] * value for index, value in features.items()) / temperature
            for row in self.weights
        ]
        return _softmax(logits)

    def predict(self, prompt: str) -> Tier:
        probabilities = self.predict_proba(prompt)
        return Tier(max(range(3), key=probabilities.__getitem__))

    @classmethod
    def train(
        cls,
        examples: Sequence[Example],
        *,
        dimensions: int = 1024,
        epochs: int = 35,
        learning_rate: float = 0.35,
        l2: float = 1e-5,
        seed: int = 17,
        underroute_weight: float = 2.0,
    ) -> "RouterModel":
        if not examples:
            raise ValueError("training examples cannot be empty")
        if epochs < 1 or learning_rate <= 0 or l2 < 0 or underroute_weight < 1:
            raise ValueError("invalid training hyperparameters")
        model = cls.empty(dimensions)
        rng = random.Random(seed)
        order = list(range(len(examples)))

        for epoch in range(epochs):
            rng.shuffle(order)
            rate = learning_rate / math.sqrt(1.0 + epoch * 0.15)
            for position in order:
                example = examples[position]
                features = extract_features(example.prompt, dimensions)
                probabilities = model.predict_proba(example.prompt)
                # Mistaking medium/high requirements for low is progressively more costly.
                sample_weight = 1.0 + underroute_weight * int(example.label)
                for class_index, row in enumerate(model.weights):
                    error = (probabilities[class_index] - int(class_index == example.label)) * sample_weight
                    for feature_index, value in features.items():
                        row[feature_index] -= rate * (error * value + l2 * row[feature_index])
        return model

    def to_dict(self) -> dict[str, object]:
        core: dict[str, object] = {
            "format": MODEL_FORMAT,
            "dimensions": self.dimensions,
            "temperature": self.temperature,
            "weights": self.weights,
        }
        canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return {**core, "sha256": hashlib.sha256(canonical).hexdigest()}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as handle:
                temporary_name = handle.name
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def load(cls, path: str | Path) -> "RouterModel":
        source = Path(path)
        try:
            if source.stat().st_size > MAX_ARTIFACT_BYTES:
                raise ArtifactError(f"model artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
            payload = json.loads(source.read_text(encoding="utf-8"))
        except ArtifactError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read model artifact {source}: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: object) -> "RouterModel":
        if not isinstance(payload, dict):
            raise ArtifactError("model artifact must be an object")
        artifact_format = payload.get("format")
        if artifact_format not in (MODEL_FORMAT, LEGACY_FORMAT):
            raise ArtifactError(f"unsupported model format: {artifact_format!r}")
        if artifact_format == MODEL_FORMAT:
            checksum = payload.get("sha256")
            core = {key: value for key, value in payload.items() if key != "sha256"}
            try:
                canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            except (TypeError, ValueError) as exc:
                raise ArtifactError(f"invalid model values: {exc}") from exc
            expected = hashlib.sha256(canonical).hexdigest()
            if not isinstance(checksum, str) or checksum != expected:
                raise ArtifactError("model artifact checksum mismatch")
        try:
            dimensions = int(payload["dimensions"])
            weights = payload["weights"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("model artifact is missing valid dimensions or weights") from exc
        if (
            dimensions < 8
            or not isinstance(weights, list)
            or len(weights) != 3
            or any(not isinstance(row, list) or len(row) != dimensions for row in weights)
        ):
            raise ArtifactError("invalid model dimensions")
        try:
            clean_weights = [[float(value) for value in row] for row in weights]
        except (TypeError, ValueError) as exc:
            raise ArtifactError("model contains non-numeric weights") from exc
        if any(not math.isfinite(value) for row in clean_weights for value in row):
            raise ArtifactError("model contains non-finite weights")
        try:
            temperature = float(payload.get("temperature", 1.0))
        except (TypeError, ValueError) as exc:
            raise ArtifactError("model temperature must be numeric") from exc
        if not math.isfinite(temperature) or temperature <= 0:
            raise ArtifactError("model temperature must be finite and positive")
        return cls(dimensions, clean_weights, temperature)
