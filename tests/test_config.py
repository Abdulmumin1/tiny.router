import json
import tempfile
import unittest
from pathlib import Path

from tiny_router.config import RouterConfig
from tiny_router.errors import ConfigurationError
from tiny_router.types import Tier


VALID = {
    "models": {
        "low": "vendor/tiny",
        "medium": {"model": "vendor/regular", "provider": "vendor"},
        "high": {"model": "vendor/frontier", "metadata": {"context": 100000}},
    },
    "policy": {"minimum_tier": "low", "tier_costs": [1, 5, 20]},
}


class ConfigTests(unittest.TestCase):
    def test_config_round_trip_and_lookup(self) -> None:
        config = RouterConfig.from_dict(VALID)
        self.assertEqual(config.target_for(Tier.MEDIUM).model, "vendor/regular")
        self.assertEqual(RouterConfig.from_dict(config.to_dict()).to_dict(), config.to_dict())
        with self.assertRaises(TypeError):
            config.models[Tier.LOW] = config.models[Tier.HIGH]  # type: ignore[index]
        with self.assertRaises(TypeError):
            config.target_for(Tier.HIGH).metadata["context"] = 1  # type: ignore[index]

    def test_unknown_keys_fail_fast(self) -> None:
        payload = {**VALID, "polciy": {}}
        with self.assertRaisesRegex(ConfigurationError, "polciy"):
            RouterConfig.from_dict(payload)

    def test_all_tiers_are_required(self) -> None:
        payload = {"models": {"low": "tiny", "high": "large"}}
        with self.assertRaisesRegex(ConfigurationError, "medium"):
            RouterConfig.from_dict(payload)

    def test_load_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "cannot read"):
                RouterConfig.load(path)


if __name__ == "__main__":
    unittest.main()
