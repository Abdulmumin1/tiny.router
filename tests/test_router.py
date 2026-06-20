from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from tiny_router.data import Example, label_from_scores, load_jsonl, split_examples
from tiny_router.evaluate import evaluate
from tiny_router.model import RouterModel
from tiny_router.policy import RoutingPolicy
from tiny_router.types import Tier


class DataTests(unittest.TestCase):
    def test_score_labels_choose_cheapest_acceptable_tier(self) -> None:
        self.assertEqual(label_from_scores({"low": 0.81, "medium": 0.9, "high": 1.0}, 0.8), Tier.LOW)
        self.assertEqual(label_from_scores({"low": 0.2, "medium": 0.85, "high": 0.9}, 0.8), Tier.MEDIUM)
        with self.assertRaisesRegex(ValueError, "no tier"):
            label_from_scores({"low": 0.2, "medium": 0.3, "high": 0.4}, 0.8)

    def test_scores_require_all_tiers_and_finite_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "scores.high"):
            label_from_scores({"low": 0.2, "medium": 0.9}, 0.8)
        with self.assertRaisesRegex(ValueError, "scores.low"):
            label_from_scores({"low": float("nan"), "medium": 0.9, "high": 1.0}, 0.8)

    def test_jsonl_error_has_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"prompt":"ok","label":"low"}\nnot json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad.jsonl:2"):
                load_jsonl(path)

    def test_split_is_deterministic_and_complete(self) -> None:
        examples = [Example(str(index), Tier(index % 3)) for index in range(20)]
        first = split_examples(examples, 0.25, seed=9)
        second = split_examples(examples, 0.25, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 15)
        self.assertEqual(len(first[1]), 5)
        self.assertEqual(set(first[0] + first[1]), set(examples))

    def test_split_stratifies_each_tier(self) -> None:
        examples = [Example(f"{tier.label}-{index}", tier) for tier in Tier for index in range(10)]
        training, validation = split_examples(examples, 0.2, seed=3)
        self.assertEqual([sum(item.label == tier for item in validation) for tier in Tier], [2, 2, 2])
        self.assertEqual([sum(item.label == tier for item in training) for tier in Tier], [8, 8, 8])


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = [
            Example("short rewrite extraction simple", Tier.LOW),
            Example("simple classify translate short", Tier.LOW),
            Example("implement api database explain", Tier.MEDIUM),
            Example("debug function design tests", Tier.MEDIUM),
            Example("prove theorem concurrency cryptography", Tier.HIGH),
            Example("formal derivation distributed race", Tier.HIGH),
        ]
        self.model = RouterModel.train(self.examples, dimensions=128, epochs=100, seed=3)

    def test_training_learns_small_separable_dataset(self) -> None:
        predictions = [self.model.predict(example.prompt) for example in self.examples]
        self.assertEqual(predictions, [example.label for example in self.examples])

    def test_probabilities_are_valid(self) -> None:
        probabilities = self.model.predict_proba("an entirely unseen prompt")
        self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_round_trip_preserves_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            loaded = RouterModel.load(path)
            self.assertEqual(loaded.predict_proba("prove concurrency"), self.model.predict_proba("prove concurrency"))

    def test_checksum_detects_tampering(self) -> None:
        payload = self.model.to_dict()
        payload["temperature"] = 2.0
        with self.assertRaisesRegex(ValueError, "checksum"):
            RouterModel.from_dict(payload)

    def test_loads_legacy_artifacts(self) -> None:
        payload = self.model.to_dict()
        payload["format"] = "tiny-router-v1"
        payload.pop("sha256")
        loaded = RouterModel.from_dict(payload)
        self.assertEqual(loaded.dimensions, self.model.dimensions)

    def test_load_rejects_non_finite_artifact_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            payload = {
                "format": "tiny-router-v1",
                "dimensions": 8,
                "temperature": float("nan"),
                "weights": [[0] * 8 for _ in range(3)],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "temperature"):
                RouterModel.load(path)

    def test_evaluation_counts_confusion_matrix(self) -> None:
        metrics = evaluate(self.model, self.examples, RoutingPolicy(underroute_penalty=100))
        self.assertEqual(sum(map(sum, metrics.confusion_matrix)), len(self.examples))
        self.assertGreaterEqual(metrics.accuracy, 0.5)


class PolicyTests(unittest.TestCase):
    def test_known_high_requirement_cannot_select_low_with_large_penalty(self) -> None:
        policy = RoutingPolicy(tier_costs=(1, 4, 15), underroute_penalty=100)
        costs = policy.expected_costs((0.0, 0.0, 1.0))
        self.assertEqual(Tier(min(range(3), key=costs.__getitem__)), Tier.HIGH)

    def test_zero_penalty_always_picks_cheapest(self) -> None:
        policy = RoutingPolicy(tier_costs=(1, 4, 15), underroute_penalty=0)
        self.assertEqual(policy.expected_costs((0.0, 0.0, 1.0)), (1.0, 4.0, 15.0))

    def test_decision_is_json_serializable(self) -> None:
        model = RouterModel.empty(32)
        decision = RoutingPolicy().route(model, "hello")
        json.dumps(decision.to_dict())


if __name__ == "__main__":
    unittest.main()
